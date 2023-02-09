//! Handle process life-time and message passing for proc-macro client

use std::{
    ffi::{OsStr, OsString},
    io::{self, BufRead, BufReader, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use paths::{AbsPath, AbsPathBuf};
use stdx::JodChild;

use crate::{
    msg::{Message, Request, Response, CURRENT_API_VERSION},
    ProcMacroKind, ServerError,
};

#[derive(Debug)]
pub(crate) struct ProcMacroProcessSrv {
    _process: Process,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    version: u32,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(
        process_path: AbsPathBuf,
        args: impl IntoIterator<Item = impl AsRef<OsStr>> + Clone,
    ) -> io::Result<ProcMacroProcessSrv> {
        let create_srv = |null_stderr| {
            let mut process = Process::run(process_path.clone(), args.clone(), null_stderr)?;
            let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

            io::Result::Ok(ProcMacroProcessSrv { _process: process, stdin, stdout, version: 0 })
        };
        let mut srv = create_srv(true)?;
        tracing::info!("sending version check");
        match srv.version_check() {
            Ok(v) if v > CURRENT_API_VERSION => Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "proc-macro server's api version ({}) is newer than rust-analyzer's ({})",
                    v, CURRENT_API_VERSION
                ),
            )),
            Ok(v) => {
                tracing::info!("got version {v}");
                srv = create_srv(false)?;
                srv.version = v;
                Ok(srv)
            }
            Err(e) => {
                tracing::info!(%e, "proc-macro version check failed, restarting and assuming version 0");
                create_srv(false)
            }
        }
    }

    pub(crate) fn version_check(&mut self) -> Result<u32, ServerError> {
        let request = Request::ApiVersionCheck {};
        let response = self.send_task(request)?;

        match response {
            Response::ApiVersionCheck(version) => Ok(version),
            Response::ExpandMacro { .. } | Response::ListMacros { .. } => {
                Err(ServerError { message: "unexpected response".to_string(), io: None })
            }
        }
    }

    pub(crate) fn find_proc_macros(
        &mut self,
        dylib_path: &AbsPath,
    ) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
        let request = Request::ListMacros { dylib_path: dylib_path.to_path_buf().into() };

        let response = self.send_task(request)?;

        match response {
            Response::ListMacros(it) => Ok(it),
            Response::ExpandMacro { .. } | Response::ApiVersionCheck { .. } => {
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
        null_stderr: bool,
    ) -> io::Result<Process> {
        let args: Vec<OsString> = args.into_iter().map(|s| s.as_ref().into()).collect();
        let child = JodChild(mk_child(&path, args, null_stderr)?);
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
    null_stderr: bool,
) -> io::Result<Child> {
    Command::new(path.as_os_str())
        .args(args)
        .env("RUST_ANALYZER_INTERNALS_DO_NOT_USE", "this is unstable")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if null_stderr { Stdio::null() } else { Stdio::inherit() })
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
