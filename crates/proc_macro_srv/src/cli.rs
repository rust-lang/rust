//! Driver for proc macro server

use crate::ProcMacroSrv;
use proc_macro_api::msg::{self, Message};
use std::io;

pub fn run() -> io::Result<()> {
    let mut srv = ProcMacroSrv::default();
    let mut buf = String::new();

    while let Some(req) = read_request(&mut buf)? {
        let res = match req {
            msg::Request::ListMacro(task) => srv.list_macros(&task).map(msg::Response::ListMacro),
            msg::Request::ExpansionMacro(task) => {
                srv.expand(&task).map(msg::Response::ExpansionMacro)
            }
        };

        let msg = res.unwrap_or_else(|err| {
            msg::Response::Error(msg::ResponseError {
                code: msg::ErrorCode::ExpansionError,
                message: err,
            })
        });

        if let Err(err) = write_response(msg) {
            eprintln!("Write message error: {}", err);
        }
    }

    Ok(())
}

fn read_request(buf: &mut String) -> io::Result<Option<msg::Request>> {
    msg::Request::read(&mut io::stdin().lock(), buf)
}

fn write_response(msg: msg::Response) -> io::Result<()> {
    msg.write(&mut io::stdout().lock())
}
