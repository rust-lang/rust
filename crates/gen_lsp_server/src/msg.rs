use std::io::{BufRead, Write};

use languageserver_types::{notification::Notification, request::Request};
use serde_derive::{Deserialize, Serialize};
use serde_json::{from_str, from_value, to_string, to_value, Value};
use failure::{bail, format_err};

use crate::Result;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum RawMessage {
    Request(RawRequest),
    Notification(RawNotification),
    Response(RawResponse),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawRequest {
    pub id: u64,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawResponse {
    // JSON RPC allows this to be null if it was impossible
    // to decode the request's id. Ignore this special case
    // and just die horribly.
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RawResponseError>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawResponseError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Copy, Debug)]
#[allow(unused)]
pub enum ErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    ServerErrorStart = -32099,
    ServerErrorEnd = -32000,
    ServerNotInitialized = -32002,
    UnknownErrorCode = -32001,
    RequestCancelled = -32800,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RawNotification {
    pub method: String,
    pub params: Value,
}

impl RawMessage {
    pub fn read(r: &mut impl BufRead) -> Result<Option<RawMessage>> {
        let text = match read_msg_text(r)? {
            None => return Ok(None),
            Some(text) => text,
        };
        let msg = from_str(&text)?;
        Ok(Some(msg))
    }
    pub fn write(self, w: &mut impl Write) -> Result<()> {
        #[derive(Serialize)]
        struct JsonRpc {
            jsonrpc: &'static str,
            #[serde(flatten)]
            msg: RawMessage,
        }
        let text = to_string(&JsonRpc {
            jsonrpc: "2.0",
            msg: self,
        })?;
        write_msg_text(w, &text)?;
        Ok(())
    }
}

impl RawRequest {
    pub fn new<R>(id: u64, params: &R::Params) -> RawRequest
    where
        R: Request,
        R::Params: serde::Serialize,
    {
        RawRequest {
            id,
            method: R::METHOD.to_string(),
            params: to_value(params).unwrap(),
        }
    }
    pub fn cast<R>(self) -> ::std::result::Result<(u64, R::Params), RawRequest>
    where
        R: Request,
        R::Params: serde::de::DeserializeOwned,
    {
        if self.method != R::METHOD {
            return Err(self);
        }
        let id = self.id;
        let params: R::Params = from_value(self.params).unwrap();
        Ok((id, params))
    }
}

impl RawResponse {
    pub fn ok<R>(id: u64, result: &R::Result) -> RawResponse
    where
        R: Request,
        R::Result: serde::Serialize,
    {
        RawResponse {
            id,
            result: Some(to_value(&result).unwrap()),
            error: None,
        }
    }
    pub fn err(id: u64, code: i32, message: String) -> RawResponse {
        let error = RawResponseError {
            code,
            message,
            data: None,
        };
        RawResponse {
            id,
            result: None,
            error: Some(error),
        }
    }
}

impl RawNotification {
    pub fn new<N>(params: &N::Params) -> RawNotification
    where
        N: Notification,
        N::Params: serde::Serialize,
    {
        RawNotification {
            method: N::METHOD.to_string(),
            params: to_value(params).unwrap(),
        }
    }
    pub fn cast<N>(self) -> ::std::result::Result<N::Params, RawNotification>
    where
        N: Notification,
        N::Params: serde::de::DeserializeOwned,
    {
        if self.method != N::METHOD {
            return Err(self);
        }
        Ok(from_value(self.params).unwrap())
    }
}

fn read_msg_text(inp: &mut impl BufRead) -> Result<Option<String>> {
    let mut size = None;
    let mut buf = String::new();
    loop {
        buf.clear();
        if inp.read_line(&mut buf)? == 0 {
            return Ok(None);
        }
        if !buf.ends_with("\r\n") {
            bail!("malformed header: {:?}", buf);
        }
        let buf = &buf[..buf.len() - 2];
        if buf.is_empty() {
            break;
        }
        let mut parts = buf.splitn(2, ": ");
        let header_name = parts.next().unwrap();
        let header_value = parts
            .next()
            .ok_or_else(|| format_err!("malformed header: {:?}", buf))?;
        if header_name == "Content-Length" {
            size = Some(header_value.parse::<usize>()?);
        }
    }
    let size = size.ok_or_else(|| format_err!("no Content-Length"))?;
    let mut buf = buf.into_bytes();
    buf.resize(size, 0);
    inp.read_exact(&mut buf)?;
    let buf = String::from_utf8(buf)?;
    log::debug!("< {}", buf);
    Ok(Some(buf))
}

fn write_msg_text(out: &mut impl Write, msg: &str) -> Result<()> {
    log::debug!("> {}", msg);
    write!(out, "Content-Length: {}\r\n\r\n", msg.len())?;
    out.write_all(msg.as_bytes())?;
    out.flush()?;
    Ok(())
}
