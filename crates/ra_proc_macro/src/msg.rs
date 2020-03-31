//! Defines messages for cross-process message based on `ndjson` wire protocol

use std::{
    convert::TryFrom,
    io::{self, BufRead, Write},
};

use crate::{
    rpc::{ListMacrosResult, ListMacrosTask},
    ExpansionResult, ExpansionTask,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Request {
    ListMacro(ListMacrosTask),
    ExpansionMacro(ExpansionTask),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Response {
    Error(ResponseError),
    ListMacro(ListMacrosResult),
    ExpansionMacro(ExpansionResult),
}

macro_rules! impl_try_from_response {
    ($ty:ty, $tag:ident) => {
        impl TryFrom<Response> for $ty {
            type Error = &'static str;
            fn try_from(value: Response) -> Result<Self, Self::Error> {
                match value {
                    Response::$tag(res) => Ok(res),
                    _ => Err("Fail to convert from response"),
                }
            }
        }
    };
}

impl_try_from_response!(ListMacrosResult, ListMacro);
impl_try_from_response!(ExpansionResult, ExpansionMacro);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResponseError {
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ErrorCode {
    ServerErrorEnd,
    ExpansionError,
}

pub trait Message: Sized + Serialize + DeserializeOwned {
    fn read(r: &mut impl BufRead) -> io::Result<Option<Self>> {
        let text = match read_json(r)? {
            None => return Ok(None),
            Some(text) => text,
        };
        let msg = serde_json::from_str(&text)?;
        Ok(Some(msg))
    }
    fn write(self, w: &mut impl Write) -> io::Result<()> {
        let text = serde_json::to_string(&self)?;
        write_json(w, &text)
    }
}

impl Message for Request {}
impl Message for Response {}

fn read_json(inp: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut buf = String::new();
    if inp.read_line(&mut buf)? == 0 {
        return Ok(None);
    }
    // Remove ending '\n'
    let buf = &buf[..buf.len() - 1];
    if buf.is_empty() {
        return Ok(None);
    }
    Ok(Some(buf.to_string()))
}

fn write_json(out: &mut impl Write, msg: &str) -> io::Result<()> {
    log::debug!("> {}", msg);
    out.write_all(msg.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
