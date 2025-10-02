// netlify/functions/api.js
// API minimale avec CORS et quelques routes de test

export const handler = async (event) => {
  const origin = event.headers?.origin || "*";
  const cors = {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  };

  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 200, headers: cors, body: "" };
  }

  const path = (event.path || "").replace(/^\/api\/?/, "").replace(/^\.netlify\/functions\/api\/?/, "");

  try {
    if (event.httpMethod === "GET" && (path === "" || path === "hello")) {
      return json({ ok: true, message: "API Netlify opérationnelle", time: new Date().toISOString() }, cors);
    }

    if (event.httpMethod === "GET" && path === "sum") {
      const a = parseFloat(event.queryStringParameters?.a ?? "0");
      const b = parseFloat(event.queryStringParameters?.b ?? "0");
      return json({ a, b, sum: a + b }, cors);
    }

    if (event.httpMethod === "POST" && path === "echo") {
      const body = safeJson(event.body);
      return json({ ok: true, you_sent: body }, cors);
    }

    if (event.httpMethod === "GET" && path === "env") {
      const secret = process.env.MY_SECRET || null;
      return json({ MY_SECRET: secret ? "(set)" : null }, cors);
    }

    return json({ error: "Route not found", path }, cors, 404);
  } catch (e) {
    return json({ error: e.message, stack: e.stack }, cors, 500);
  }
};

function json(data, headers = {}, statusCode = 200) {
  return {
    statusCode,
    headers: { "Content-Type": "application/json; charset=utf-8", ...headers },
    body: JSON.stringify(data),
  };
}
function safeJson(body) {
  try { return body ? JSON.parse(body) : {}; } catch { return { _raw: body }; }
}