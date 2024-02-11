//! Handle process life-time and message passing for proc-macro client

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};

use paths::{AbsPath, AbsPathBuf};
use stdx::JodChild;

use crate::{
    msg::{Message, Request, Response, SpanMode, CURRENT_API_VERSION, RUST_ANALYZER_SPAN_SUPPORT},
    ProcMacroKind, ServerError,
};

#[derive(Debug)]
pub(crate) struct ProcMacroProcessSrv {
    process: Process,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Populated when the server exits.
    server_exited: Option<ServerError>,
    version: u32,
    mode: SpanMode,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(process_path: AbsPathBuf) -> io::Result<ProcMacroProcessSrv> {
        let create_srv = |null_stderr| {
            let mut process = Process::run(process_path.clone(), null_stderr)?;
            let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

            io::Result::Ok(ProcMacroProcessSrv {
                process,
                stdin,
                stdout,
                server_exited: None,
                version: 0,
                mode: SpanMode::Id,
            })
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
                if srv.version > RUST_ANALYZER_SPAN_SUPPORT {
                    if let Ok(mode) = srv.enable_rust_analyzer_spans() {
                        srv.mode = mode;
                    }
                }
                Ok(srv)
            }
            Err(e) => {
                tracing::info!(%e, "proc-macro version check failed, restarting and assuming version 0");
                create_srv(false)
            }
        }
    }

    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    pub(crate) fn version_check(&mut self) -> Result<u32, ServerError> {
        let request = Request::ApiVersionCheck {};
        let response = self.send_task(request)?;

        match response {
            Response::ApiVersionCheck(version) => Ok(version),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    fn enable_rust_analyzer_spans(&mut self) -> Result<SpanMode, ServerError> {
        let request = Request::SetConfig(crate::msg::ServerConfig {
            span_mode: crate::msg::SpanMode::RustAnalyzer,
        });
        let response = self.send_task(request)?;

        match response {
            Response::SetConfig(crate::msg::ServerConfig { span_mode }) => Ok(span_mode),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
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
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    pub(crate) fn send_task(&mut self, req: Request) -> Result<Response, ServerError> {
        if let Some(server_error) = &self.server_exited {
            return Err(server_error.clone());
        }

        let mut buf = String::new();
        send_request(&mut self.stdin, &mut self.stdout, req, &mut buf).map_err(|e| {
            if e.io.as_ref().map(|it| it.kind()) == Some(io::ErrorKind::BrokenPipe) {
                match self.process.child.try_wait() {
                    Ok(None) => e,
                    Ok(Some(status)) => {
                        let mut msg = String::new();
                        if !status.success() {
                            if let Some(stderr) = self.process.child.stderr.as_mut() {
                                _ = stderr.read_to_string(&mut msg);
                            }
                        }
                        let server_error = ServerError {
                            message: format!("server exited with {status}: {msg}"),
                            io: None,
                        };
                        self.server_exited = Some(server_error.clone());
                        server_error
                    }
                    Err(_) => e,
                }
            } else {
                e
            }
        })
    }
}

#[derive(Debug)]
struct Process {
    child: JodChild,
}

impl Process {
    fn run(path: AbsPathBuf, null_stderr: bool) -> io::Result<Process> {
        let child = JodChild(mk_child(&path, null_stderr)?);
        Ok(Process { child })
    }

    fn stdio(&mut self) -> Option<(ChildStdin, BufReader<ChildStdout>)> {
        let stdin = self.child.stdin.take()?;
        let stdout = self.child.stdout.take()?;
        let read = BufReader::new(stdout);

        Some((stdin, read))
    }
}

fn mk_child(path: &AbsPath, null_stderr: bool) -> io::Result<Child> {
    let mut cmd = Command::new(path.as_os_str());
    cmd.env("RUST_ANALYZER_INTERNALS_DO_NOT_USE", "this is unstable")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if null_stderr { Stdio::null() } else { Stdio::inherit() });
    if cfg!(windows) {
        let mut path_var = std::ffi::OsString::new();
        path_var.push(path.parent().unwrap().parent().unwrap().as_os_str());
        path_var.push("\\bin;");
        path_var.push(std::env::var_os("PATH").unwrap_or_default());
        cmd.env("PATH", path_var);
    }
    cmd.spawn()
}

fn send_request(
    mut writer: &mut impl Write,
    mut reader: &mut impl BufRead,
    req: Request,
    buf: &mut String,
) -> Result<Response, ServerError> {
    req.write(&mut writer).map_err(|err| ServerError {
        message: "failed to write request".into(),
        io: Some(Arc::new(err)),
    })?;
    let res = Response::read(&mut reader, buf).map_err(|err| ServerError {
        message: "failed to read response".into(),
        io: Some(Arc::new(err)),
    })?;
    res.ok_or_else(|| ServerError { message: "server exited".into(), io: None })
}
