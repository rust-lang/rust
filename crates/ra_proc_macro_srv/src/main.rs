use ra_proc_macro::msg::{self, Message};
use ra_proc_macro_srv::{expand_task, list_macros};

use std::io;

fn read_request() -> Result<Option<msg::Request>, io::Error> {
    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    msg::Request::read(&mut stdin)
}

fn write_response(res: Result<msg::Response, String>) -> Result<(), io::Error> {
    let msg: msg::Response = match res {
        Ok(res) => res,
        Err(err) => msg::Response::Error(msg::ResponseError {
            code: msg::ErrorCode::ExpansionError,
            message: err,
        }),
    };

    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    msg.write(&mut stdout)
}
fn main() {
    loop {
        let req = match read_request() {
            Err(err) => {
                eprintln!("Read message error on ra_proc_macro_srv: {}", err.to_string());
                continue;
            }
            Ok(None) => continue,
            Ok(Some(req)) => req,
        };

        match req {
            msg::Request::ListMacro(task) => {
                if let Err(err) =
                    write_response(list_macros(&task).map(|it| msg::Response::ListMacro(it)))
                {
                    eprintln!("Write message error on list macro: {}", err);
                }
            }
            msg::Request::ExpansionMacro(task) => {
                if let Err(err) =
                    write_response(expand_task(&task).map(|it| msg::Response::ExpansionMacro(it)))
                {
                    eprintln!("Write message error on expansion macro: {}", err);
                }
            }
        }
    }
}
