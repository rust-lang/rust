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
    fn read(inp: &mut impl BufRead, buf: &mut String) -> io::Result<Option<Self>> {
        Ok(match read_json(inp, buf)? {
            None => None,
            Some(text) => {
                let mut deserializer = serde_json::Deserializer::from_str(&text);
                // Note that some proc-macro generate very deep syntax tree
                // We have to disable the current limit of serde here
                deserializer.disable_recursion_limit();
                Some(Self::deserialize(&mut deserializer)?)
            }
        })
    }
    fn write(self, out: &mut impl Write) -> io::Result<()> {
        let text = serde_json::to_string(&self)?;
        write_json(out, &text)
    }
}

impl Message for Request {}
impl Message for Response {}

fn read_json<'a>(
    inp: &mut impl BufRead,
    mut buf: &'a mut String,
) -> io::Result<Option<&'a String>> {
    loop {
        buf.clear();

        inp.read_line(&mut buf)?;
        buf.pop(); // Remove trailing '\n'

        if buf.is_empty() {
            return Ok(None);
        }

        // Some ill behaved macro try to use stdout for debugging
        // We ignore it here
        if !buf.starts_with('{') {
            log::error!("proc-macro tried to print : {}", buf);
            continue;
        }

        return Ok(Some(buf));
    }
}

fn write_json(out: &mut impl Write, msg: &str) -> io::Result<()> {
    log::debug!("> {}", msg);
    out.write_all(msg.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}
