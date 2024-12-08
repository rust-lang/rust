//! Handle process life-time and message passing for proc-macro client

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    panic::AssertUnwindSafe,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
};

use paths::AbsPath;
use stdx::JodChild;

use crate::{
    json::{read_json, write_json},
    msg::{Message, Request, Response, SpanMode, CURRENT_API_VERSION, RUST_ANALYZER_SPAN_SUPPORT},
    ProcMacroKind, ServerError,
};

#[derive(Debug)]
pub(crate) struct ProcMacroProcessSrv {
    /// The state of the proc-macro server process, the protocol is currently strictly sequential
    /// hence the lock on the state.
    state: Mutex<ProcessSrvState>,
    version: u32,
    mode: SpanMode,
    /// Populated when the server exits.
    exited: OnceLock<AssertUnwindSafe<ServerError>>,
}

#[derive(Debug)]
struct ProcessSrvState {
    process: Process,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ProcMacroProcessSrv {
    pub(crate) fn run(
        process_path: &AbsPath,
        env: impl IntoIterator<Item = (impl AsRef<std::ffi::OsStr>, impl AsRef<std::ffi::OsStr>)>
            + Clone,
    ) -> io::Result<ProcMacroProcessSrv> {
        let create_srv = |null_stderr| {
            let mut process = Process::run(process_path, env.clone(), null_stderr)?;
            let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

            io::Result::Ok(ProcMacroProcessSrv {
                state: Mutex::new(ProcessSrvState { process, stdin, stdout }),
                version: 0,
                mode: SpanMode::Id,
                exited: OnceLock::new(),
            })
        };
        let mut srv = create_srv(true)?;
        tracing::info!("sending proc-macro server version check");
        match srv.version_check() {
            Ok(v) if v > CURRENT_API_VERSION => Err(io::Error::new(
                io::ErrorKind::Other,
                format!( "The version of the proc-macro server ({v}) in your Rust toolchain is newer than the version supported by your rust-analyzer ({CURRENT_API_VERSION}).
            This will prevent proc-macro expansion from working. Please consider updating your rust-analyzer to ensure compatibility with your current toolchain."
                ),
            )),
            Ok(v) => {
                tracing::info!("Proc-macro server version: {v}");
                srv = create_srv(false)?;
                srv.version = v;
                if srv.version >= RUST_ANALYZER_SPAN_SUPPORT {
                    if let Ok(mode) = srv.enable_rust_analyzer_spans() {
                        srv.mode = mode;
                    }
                }
                tracing::info!("Proc-macro server span mode: {:?}", srv.mode);
                Ok(srv)
            }
            Err(e) => {
                tracing::info!(%e, "proc-macro version check failed, restarting and assuming version 0");
                create_srv(false)
            }
        }
    }

    pub(crate) fn exited(&self) -> Option<&ServerError> {
        self.exited.get().map(|it| &it.0)
    }

    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    fn version_check(&self) -> Result<u32, ServerError> {
        let request = Request::ApiVersionCheck {};
        let response = self.send_task(request)?;

        match response {
            Response::ApiVersionCheck(version) => Ok(version),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    fn enable_rust_analyzer_spans(&self) -> Result<SpanMode, ServerError> {
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
        &self,
        dylib_path: &AbsPath,
    ) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
        let request = Request::ListMacros { dylib_path: dylib_path.to_path_buf().into() };

        let response = self.send_task(request)?;

        match response {
            Response::ListMacros(it) => Ok(it),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    pub(crate) fn send_task(&self, req: Request) -> Result<Response, ServerError> {
        if let Some(server_error) = self.exited.get() {
            return Err(server_error.0.clone());
        }

        let state = &mut *self.state.lock().unwrap();
        let mut buf = String::new();
        send_request(&mut state.stdin, &mut state.stdout, req, &mut buf)
            .and_then(|res| {
                res.ok_or_else(|| {
                    let message = "proc-macro server did not respond with data".to_owned();
                    ServerError {
                        io: Some(Arc::new(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            message.clone(),
                        ))),
                        message,
                    }
                })
            })
            .map_err(|e| {
                if e.io.as_ref().map(|it| it.kind()) == Some(io::ErrorKind::BrokenPipe) {
                    match state.process.child.try_wait() {
                        Ok(None) | Err(_) => e,
                        Ok(Some(status)) => {
                            let mut msg = String::new();
                            if !status.success() {
                                if let Some(stderr) = state.process.child.stderr.as_mut() {
                                    _ = stderr.read_to_string(&mut msg);
                                }
                            }
                            let server_error = ServerError {
                                message: format!(
                                    "proc-macro server exited with {status}{}{msg}",
                                    if msg.is_empty() { "" } else { ": " }
                                ),
                                io: None,
                            };
                            // `AssertUnwindSafe` is fine here, we already correct initialized
                            // server_error at this point.
                            self.exited.get_or_init(|| AssertUnwindSafe(server_error)).0.clone()
                        }
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
    fn run(
        path: &AbsPath,
        env: impl IntoIterator<Item = (impl AsRef<std::ffi::OsStr>, impl AsRef<std::ffi::OsStr>)>,
        null_stderr: bool,
    ) -> io::Result<Process> {
        let child = JodChild(mk_child(path, env, null_stderr)?);
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
    env: impl IntoIterator<Item = (impl AsRef<std::ffi::OsStr>, impl AsRef<std::ffi::OsStr>)>,
    null_stderr: bool,
) -> io::Result<Child> {
    let mut cmd = Command::new(path);
    cmd.envs(env)
        .env("RUST_ANALYZER_INTERNALS_DO_NOT_USE", "this is unstable")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(if null_stderr { Stdio::null() } else { Stdio::inherit() });
    if cfg!(windows) {
        let mut path_var = std::ffi::OsString::new();
        path_var.push(path.parent().unwrap().parent().unwrap());
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
) -> Result<Option<Response>, ServerError> {
    req.write(write_json, &mut writer).map_err(|err| ServerError {
        message: "failed to write request".into(),
        io: Some(Arc::new(err)),
    })?;
    let res = Response::read(read_json, &mut reader, buf).map_err(|err| ServerError {
        message: "failed to read response".into(),
        io: Some(Arc::new(err)),
    })?;
    Ok(res)
}
