#![feature(restricted_std)]

use std::fmt::Write as FmtWrite;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::task::{Context, Poll};

use serde::{Deserialize, Serialize};

use llm::{ChatDelta, ChatRequest, ChatStream, FinishReason, LlmError, Role, StreamingLlmClient};

// ── Ollama API wire types ────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: Option<OllamaResponseMessage>,
    done: bool,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    content: String,
}

// ── Public client ────────────────────────────────────────────────────

pub struct OllamaClient {
    base_url: String,
    model: String,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }
}

impl StreamingLlmClient for OllamaClient {
    fn chat_stream(&self, req: ChatRequest) -> Result<Box<dyn ChatStream + Send>, LlmError> {
        let messages: Vec<OllamaMessage> = req
            .messages
            .iter()
            .map(|m| OllamaMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let ollama_req = OllamaRequest {
            model: &self.model,
            messages,
            stream: true,
        };

        let body = serde_json::to_string(&ollama_req)
            .map_err(|_| LlmError::Other("Serialize error".to_string()))?;
        let url = format!("{}/api/chat", self.base_url);

        let (stream, leftover) = http_post(&url, &body).map_err(LlmError::Transport)?;

        Ok(Box::new(OllamaChatStream {
            stream,
            buffer: leftover,
            done: false,
        }))
    }
}

// ── Streaming response reader ────────────────────────────────────────

struct OllamaChatStream {
    stream: TcpStream,
    buffer: Vec<u8>,
    done: bool,
}

impl ChatStream for OllamaChatStream {
    fn poll_next(&mut self, _cx: &mut Context<'_>) -> Poll<Result<Option<ChatDelta>, LlmError>> {
        if self.done && self.buffer.is_empty() {
            return Poll::Ready(Ok(None));
        }

        // Try to read more data
        let mut tmp = [0u8; 4096];
        match self.stream.read(&mut tmp) {
            Ok(0) => {
                if self.buffer.is_empty() {
                    return Poll::Ready(Ok(None));
                }
            }
            Ok(n) => {
                self.buffer.extend_from_slice(&tmp[..n]);
            }
            Err(e) => return Poll::Ready(Err(LlmError::Transport(e.to_string()))),
        }

        let mut consumed = 0;
        let mut result = None;

        {
            let mut iter =
                serde_json::Deserializer::from_slice(&self.buffer).into_iter::<OllamaResponse>();

            while let Some(Ok(resp)) = iter.next() {
                consumed = iter.byte_offset();

                if resp.done {
                    self.done = true;
                    result = Some(ChatDelta {
                        text: String::new(),
                        finish: Some(FinishReason::Stop),
                    });
                    break;
                }

                if let Some(msg) = resp.message {
                    result = Some(ChatDelta {
                        text: msg.content,
                        finish: None,
                    });
                    break;
                }
            }
        }

        if consumed > 0 {
            self.buffer.drain(..consumed);
        }

        if let Some(delta) = result {
            return Poll::Ready(Ok(Some(delta)));
        }

        if self.done {
            return Poll::Ready(Ok(None));
        }

        Poll::Pending
    }
}

// ── Minimal HTTP POST using std::net::TcpStream ─────────────────────

/// Parse a URL into (host, port, path). For non-http:// URLs, routes
/// through the QEMU host proxy at 10.0.2.2:8081.
fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    if let Some(rest) = url.strip_prefix("http://") {
        let (host_port, path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], &rest[idx..]),
            None => (rest, "/"),
        };
        let (host, port) = match host_port.find(':') {
            Some(idx) => {
                let port = host_port[idx + 1..]
                    .parse::<u16>()
                    .map_err(|_| "Invalid port".to_string())?;
                (&host_port[..idx], port)
            }
            None => (host_port, 80),
        };
        Ok((host.to_string(), port, path.to_string()))
    } else {
        // HTTPS or other — route through QEMU host proxy
        let encoded = url_encode(url);
        let proxy_path = format!("/?url={}", encoded);
        Ok(("10.0.2.2".to_string(), 8081, proxy_path))
    }
}

/// Perform an HTTP POST and return (TcpStream, initial_body_bytes).
/// The TcpStream is positioned after the HTTP response headers.
fn http_post(url: &str, body: &str) -> Result<(TcpStream, Vec<u8>), String> {
    let (host, port, path) = parse_url(url)?;

    let addr = format!("{}:{}", host, port);
    let mut stream =
        TcpStream::connect(&addr).map_err(|e| format!("TCP connect to {}: {}", addr, e))?;

    // Build HTTP/1.1 request
    let mut req = String::new();
    write!(req, "POST {} HTTP/1.1\r\n", path).ok();
    write!(req, "Host: {}\r\n", host).ok();
    write!(req, "Connection: close\r\n").ok();
    write!(req, "Content-Length: {}\r\n", body.len()).ok();
    write!(req, "Content-Type: application/json\r\n").ok();
    write!(req, "\r\n").ok();
    req.push_str(body);

    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("HTTP write: {}", e))?;

    // Read until we find the header/body separator (\r\n\r\n)
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];

    for _ in 0..40 {
        let n = stream
            .read(&mut tmp)
            .map_err(|e| format!("HTTP read: {}", e))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);

        if find_header_end(&buf).is_some() {
            break;
        }
    }

    let body_start = find_header_end(&buf).map(|i| i + 4).unwrap_or(buf.len());
    let leftover = buf[body_start..].to_vec();

    Ok((stream, leftover))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn url_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(b) {
            out.push(*b as char);
        } else {
            write!(out, "%{:02X}", b).ok();
        }
    }
    out
}
