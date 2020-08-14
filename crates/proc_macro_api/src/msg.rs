//! Defines messages for cross-process message passing based on `ndjson` wire protocol

use std::{
    convert::TryFrom,
    io::{self, BufRead, Write},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    rpc::{ListMacrosResult, ListMacrosTask},
    ExpansionResult, ExpansionTask,
};

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
                    _ => Err(concat!("Failed to convert response to ", stringify!($tag))),
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

pub trait Message: Serialize + DeserializeOwned {
    fn read(inp: &mut impl BufRead) -> io::Result<Option<Self>> {
        Ok(match read_json(inp)? {
            None => None,
            Some(text) => Some(serde_json::from_str(&text)?),
        })
    }
    fn write(self, out: &mut impl Write) -> io::Result<()> {
        let text = serde_json::to_string(&self)?;
        write_json(out, &text)
    }
}

impl Message for Request {}
impl Message for Response {}

fn read_json(inp: &mut impl BufRead) -> io::Result<Option<String>> {
    let mut buf = String::new();
    inp.read_line(&mut buf)?;
    buf.pop(); // Remove traling '\n'
    Ok(match buf.len() {
        0 => None,
        _ => Some(buf),
    })
}

fn write_json(out: &mut impl Write, msg: &str) -> io::Result<()> {
    log::debug!("> {}", msg);
    out.write_all(msg.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
