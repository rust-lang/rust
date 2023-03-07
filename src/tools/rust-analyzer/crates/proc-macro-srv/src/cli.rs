//! Driver for proc macro server
use std::io;

use proc_macro_api::msg::{self, Message};

use crate::ProcMacroSrv;

pub fn run() -> io::Result<()> {
    let mut srv = ProcMacroSrv::default();
    let mut buf = String::new();

    while let Some(req) = read_request(&mut buf)? {
        let res = match req {
            msg::Request::ListMacros { dylib_path } => {
                msg::Response::ListMacros(srv.list_macros(&dylib_path))
            }
            msg::Request::ExpandMacro(task) => msg::Response::ExpandMacro(srv.expand(task)),
            msg::Request::ApiVersionCheck {} => {
                msg::Response::ApiVersionCheck(proc_macro_api::msg::CURRENT_API_VERSION)
            }
        };
        write_response(res)?
    }

    Ok(())
}

fn read_request(buf: &mut String) -> io::Result<Option<msg::Request>> {
    msg::Request::read(&mut io::stdin().lock(), buf)
}

fn write_response(msg: msg::Response) -> io::Result<()> {
    msg.write(&mut io::stdout().lock())
}
