//! Handle process life-time and message passing for proc-macro client

use std::{
    convert::{TryFrom, TryInto},
    ffi::{OsStr, OsString},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Mutex,
};

use stdx::JodChild;

use crate::{
    msg::{ErrorCode, Message, Request, Response, ResponseError},
    rpc::{ListMacrosResult, ListMacrosTask, ProcMacroKind},
};

#[derive(Debug)]
pub(crate) struct ProcMacroProcessSrv {
    process: Mutex<Process>,
    stdio: Mutex<(ChildStdin, BufReader<ChildStdout>)>,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(
        process_path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroProcessSrv> {
        let mut process = Process::run(process_path, args)?;
        let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

        let srv = ProcMacroProcessSrv {
            process: Mutex::new(process),
            stdio: Mutex::new((stdin, stdout)),
        };

        Ok(srv)
    }

    pub(crate) fn find_proc_macros(
        &self,
        dylib_path: &Path,
    ) -> Result<Vec<(String, ProcMacroKind)>, tt::ExpansionError> {
        let task = ListMacrosTask { lib: dylib_path.to_path_buf() };

        let result: ListMacrosResult = self.send_task(Request::ListMacro(task))?;
        Ok(result.macros)
    }

    pub(crate) fn send_task<R>(&self, req: Request) -> Result<R, tt::ExpansionError>
    where
        R: TryFrom<Response, Error = &'static str>,
    {
        let mut guard = self.stdio.lock().unwrap_or_else(|e| e.into_inner());
        let stdio = &mut *guard;
        let (stdin, stdout) = (&mut stdio.0, &mut stdio.1);

        let mut buf = String::new();
        let res = match send_request(stdin, stdout, req, &mut buf) {
            Ok(res) => res,
            Err(err) => {
                let mut process = self.process.lock().unwrap_or_else(|e| e.into_inner());
                log::error!(
                    "proc macro server crashed, server process state: {:?}, server request error: {:?}",
                    process.child.try_wait(),
                    err
                );
                let res = Response::Error(ResponseError {
                    code: ErrorCode::ServerErrorEnd,
                    message: "proc macro server crashed".into(),
                });
                Some(res)
            }
        };

        match res {
            Some(Response::Error(err)) => Err(tt::ExpansionError::ExpansionError(err.message)),
            Some(res) => Ok(res.try_into().map_err(|err| {
                tt::ExpansionError::Unknown(format!("Fail to get response, reason : {:#?} ", err))
            })?),
            None => Err(tt::ExpansionError::Unknown("Empty result".into())),
        }
    }
}

#[derive(Debug)]
struct Process {
    child: JodChild,
}

impl Process {
    fn run(
        path: PathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<Process> {
        let args: Vec<OsString> = args.into_iter().map(|s| s.as_ref().into()).collect();
        let child = JodChild(mk_child(&path, &args)?);
        Ok(Process { child })
    }

    fn stdio(&mut self) -> Option<(ChildStdin, BufReader<ChildStdout>)> {
        let stdin = self.child.stdin.take()?;
        let stdout = self.child.stdout.take()?;
        let read = BufReader::new(stdout);

        Some((stdin, read))
    }
}

fn mk_child(path: &Path, args: impl IntoIterator<Item = impl AsRef<OsStr>>) -> io::Result<Child> {
    Command::new(&path)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
}

fn send_request(
    mut writer: &mut impl Write,
    mut reader: &mut impl BufRead,
    req: Request,
    buf: &mut String,
) -> io::Result<Option<Response>> {
    req.write(&mut writer)?;
    Response::read(&mut reader, buf)
}
