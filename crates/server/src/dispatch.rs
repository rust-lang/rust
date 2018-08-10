use std::marker::PhantomData;

use serde::{
    ser::Serialize,
    de::DeserializeOwned,
};
use serde_json;
use drop_bomb::DropBomb;

use ::{
    Result,
    req::{Request, Notification},
    io::{Io, RawMsg, RawResponse, RawRequest, RawNotification},
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
    pub fn response(self, io: &mut Io, resp: Result<R::Result>) -> Result<()> {
        match resp {
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


fn parse_request_as<R>(raw: RawRequest) -> Result<::std::result::Result<(R::Params, Responder<R>), RawRequest>>
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

pub fn handle_request<R, F>(req: &mut Option<RawRequest>, f: F) -> Result<()>
    where
        R: Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
        F: FnOnce(R::Params, Responder<R>) -> Result<()>
{
    match req.take() {
        None => Ok(()),
        Some(r) => match parse_request_as::<R>(r)? {
            Ok((params, responder)) => f(params, responder),
            Err(r) => {
                *req = Some(r);
                Ok(())
            },
        }
    }
}

pub fn expect_request<R>(io: &mut Io, raw: RawRequest) -> Result<Option<(R::Params, Responder<R>)>>
    where
        R: Request,
        R::Params: DeserializeOwned,
        R::Result: Serialize,
{
    let ret = match parse_request_as::<R>(raw)? {
        Ok(x) => Some(x),
        Err(raw) => {
            unknown_method(io, raw)?;
            None
        }
    };
    Ok(ret)
}

fn parse_notification_as<N>(raw: RawNotification) -> Result<::std::result::Result<N::Params, RawNotification>>
    where
        N: Notification,
        N::Params: DeserializeOwned,
{
    if raw.method != N::METHOD {
        return Ok(Err(raw));
    }
    let params: N::Params = serde_json::from_value(raw.params)?;
    Ok(Ok(params))
}

pub fn handle_notification<N, F>(not: &mut Option<RawNotification>, f: F) -> Result<()>
    where
        N: Notification,
        N::Params: DeserializeOwned,
        F: FnOnce(N::Params) -> Result<()>
{
    match not.take() {
        None => Ok(()),
        Some(n) => match parse_notification_as::<N>(n)? {
            Ok(params) => f(params),
            Err(n) => {
                *not = Some(n);
                Ok(())
            },
        }
    }
}

pub fn send_notification<N>(io: &mut Io, params: N::Params) -> Result<()>
    where
        N: Notification,
        N::Params: Serialize
{
    io.send(RawMsg::Notification(RawNotification {
        method: N::METHOD.to_string(),
        params: serde_json::to_value(params)?,
    }));
    Ok(())
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
