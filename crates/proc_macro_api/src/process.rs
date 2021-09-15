//! Handle process life-time and message passing for proc-macro client

use std::{
    ffi::{OsStr, OsString},
    io::{self, BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use paths::{AbsPath, AbsPathBuf};
use stdx::JodChild;

use crate::{
    msg::{Message, Request, Response},
    ProcMacroKind, ServerError,
};

#[derive(Debug)]
pub(crate) struct ProcMacroProcessSrv {
    _process: Process,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(
        process_path: AbsPathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> io::Result<ProcMacroProcessSrv> {
        let mut process = Process::run(process_path, args)?;
        let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

        let srv = ProcMacroProcessSrv { _process: process, stdin, stdout };

        Ok(srv)
    }

    pub(crate) fn find_proc_macros(
        &mut self,
        dylib_path: &AbsPath,
    ) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
        let request = Request::ListMacros { dylib_path: dylib_path.to_path_buf().into() };

        let response = self.send_task(request)?;

        match response {
            Response::ListMacros(it) => Ok(it),
            Response::ExpandMacro { .. } => {
                Err(ServerError { message: "unexpected response".to_string(), io: None })
            }
        }
    }

    pub(crate) fn send_task(&mut self, req: Request) -> Result<Response, ServerError> {
        let mut buf = String::new();
        send_request(&mut self.stdin, &mut self.stdout, req, &mut buf)
    }
}

#[derive(Debug)]
struct Process {
    child: JodChild,
}

impl Process {
    fn run(
        path: AbsPathBuf,
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

fn mk_child(
    path: &AbsPath,
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> io::Result<Child> {
    Command::new(path.as_os_str())
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
) -> Result<Response, ServerError> {
    req.write(&mut writer)
        .map_err(|err| ServerError { message: "failed to write request".into(), io: Some(err) })?;
    let res = Response::read(&mut reader, buf)
        .map_err(|err| ServerError { message: "failed to read response".into(), io: Some(err) })?;
    res.ok_or_else(|| ServerError { message: "server exited".into(), io: None })
}
