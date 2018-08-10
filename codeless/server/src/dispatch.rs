use std::marker::PhantomData;

use serde::{
    ser::Serialize,
    de::DeserializeOwned,
};
use serde_json;
use drop_bomb::DropBomb;

use ::{
    Result,
    req::Request,
    io::{Io, RawMsg, RawResponse, RawRequest},
};

pub struct Responder<R: Request> {
    id: u64,
    bomb: DropBomb,
    ph: PhantomData<R>,
}

impl<R: Request> Responder<R>
    where
        R::Params: DeserializeOwned,
        R::Result: Serialize,
{
    pub fn respond_with(self, io: &mut Io, f: impl FnOnce() -> Result<R::Result>) -> Result<()> {
        match f() {
            Ok(res) => self.result(io, res)?,
            Err(e) => {
                self.error(io)?;
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn result(mut self, io: &mut Io, result: R::Result) -> Result<()> {
        self.bomb.defuse();
        io.send(RawMsg::Response(RawResponse {
            id: Some(self.id),
            result: serde_json::to_value(result)?,
            error: serde_json::Value::Null,
        }));
        Ok(())
    }

    pub fn error(mut self, io: &mut Io) -> Result<()> {
        self.bomb.defuse();
        error(io, self.id, ErrorCode::InternalError, "internal error")
    }
}


pub fn parse_as<R>(raw: RawRequest) -> Result<::std::result::Result<(R::Params, Responder<R>), RawRequest>>
    where
        R: Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
{
    if raw.method != R::METHOD {
        return Ok(Err(raw));
    }

    let params: R::Params = serde_json::from_value(raw.params)?;
    let responder = Responder {
        id: raw.id,
        bomb: DropBomb::new("dropped request"),
        ph: PhantomData,
    };
    Ok(Ok((params, responder)))
}

pub fn expect<R>(io: &mut Io, raw: RawRequest) -> Result<Option<(R::Params, Responder<R>)>>
    where
        R: Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
{
    let ret = match parse_as::<R>(raw)? {
        Ok(x) => Some(x),
        Err(raw) => {
            unknown_method(io, raw)?;
            None
        }
    };
    Ok(ret)
}

pub fn unknown_method(io: &mut Io, raw: RawRequest) -> Result<()> {
    error(io, raw.id, ErrorCode::MethodNotFound, "unknown method")
}

fn error(io: &mut Io, id: u64, code: ErrorCode, message: &'static str) -> Result<()> {
    #[derive(Serialize)]
    struct Error {
        code: i32,
        message: &'static str,
    }
    io.send(RawMsg::Response(RawResponse {
        id: Some(id),
        result: serde_json::Value::Null,
        error: serde_json::to_value(Error {
            code: code as i32,
            message,
        })?,
    }));
    Ok(())
}


#[allow(unused)]
enum ErrorCode {
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
