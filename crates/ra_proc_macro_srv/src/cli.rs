//! Driver for proc macro server

use crate::{expand_task, list_macros};
use ra_proc_macro::msg::{self, Message};
use std::io;

pub fn run() {
    loop {
        let req = match read_request() {
            Err(err) => {
                // Panic here, as the stdin pipe may be closed.
                // Otherwise, client will be restarted the service anyway.
                panic!("Read message error on ra_proc_macro_srv: {}", err);
            }
            Ok(None) => continue,
            Ok(Some(req)) => req,
        };

        let res = match req {
            msg::Request::ListMacro(task) => Ok(msg::Response::ListMacro(list_macros(&task))),
            msg::Request::ExpansionMacro(task) => {
                expand_task(&task).map(msg::Response::ExpansionMacro)
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
}

fn read_request() -> io::Result<Option<msg::Request>> {
    msg::Request::read(&mut io::stdin().lock())
}

fn write_response(msg: msg::Response) -> io::Result<()> {
    msg.write(&mut io::stdout().lock())
}
