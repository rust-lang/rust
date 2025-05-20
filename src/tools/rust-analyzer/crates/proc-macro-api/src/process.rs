//! Handle process life-time and message passing for proc-macro client

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    panic::AssertUnwindSafe,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
};

use paths::AbsPath;
use span::FIXUP_ERASED_FILE_AST_ID_MARKER;
use stdx::JodChild;

use crate::{
    ProcMacroKind, ServerError,
    legacy_protocol::{
        json::{read_json, write_json},
        msg::{
            CURRENT_API_VERSION, HASHED_AST_ID, Message, Request, Response, ServerConfig, SpanMode,
        },
    },
};

/// Represents a process handling proc-macro communication.
#[derive(Debug)]
pub(crate) struct ProcMacroServerProcess {
    /// The state of the proc-macro server process, the protocol is currently strictly sequential
    /// hence the lock on the state.
    state: Mutex<ProcessSrvState>,
    version: u32,
    mode: SpanMode,
    /// Populated when the server exits.
    exited: OnceLock<AssertUnwindSafe<ServerError>>,
}

/// Maintains the state of the proc-macro server process.
#[derive(Debug)]
struct ProcessSrvState {
    process: Process,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ProcMacroServerProcess {
    /// Starts the proc-macro server and performs a version check
    pub(crate) fn run<'a>(
        process_path: &AbsPath,
        env: impl IntoIterator<
            Item = (impl AsRef<std::ffi::OsStr>, &'a Option<impl 'a + AsRef<std::ffi::OsStr>>),
        > + Clone,
    ) -> io::Result<ProcMacroServerProcess> {
        let create_srv = || {
            let mut process = Process::run(process_path, env.clone())?;
            let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

            io::Result::Ok(ProcMacroServerProcess {
                state: Mutex::new(ProcessSrvState { process, stdin, stdout }),
                version: 0,
                mode: SpanMode::Id,
                exited: OnceLock::new(),
            })
        };
        let mut srv = create_srv()?;
        tracing::info!("sending proc-macro server version check");
        match srv.version_check() {
            Ok(v) if v > CURRENT_API_VERSION => Err(io::Error::other(
                format!( "The version of the proc-macro server ({v}) in your Rust toolchain is newer than the version supported by your rust-analyzer ({CURRENT_API_VERSION}).
            This will prevent proc-macro expansion from working. Please consider updating your rust-analyzer to ensure compatibility with your current toolchain."
                ),
            )),
            Ok(v) => {
                tracing::info!("Proc-macro server version: {v}");
                srv.version = v;
                if srv.version >= HASHED_AST_ID {
                    // We don't enable spans on versions prior to `HASHED_AST_ID`, because their ast id layout
                    // is different.
                    if let Ok(mode) = srv.enable_rust_analyzer_spans() {
                        srv.mode = mode;
                    }
                }
                tracing::info!("Proc-macro server span mode: {:?}", srv.mode);
                Ok(srv)
            }
            Err(e) => {
                tracing::info!(%e, "proc-macro version check failed");
                Err(
                    io::Error::other(format!("proc-macro server version check failed: {e}")),
                )
            }
        }
    }

    /// Returns the server error if the process has exited.
    pub(crate) fn exited(&self) -> Option<&ServerError> {
        self.exited.get().map(|it| &it.0)
    }

    /// Retrieves the API version of the proc-macro server.
    pub(crate) fn version(&self) -> u32 {
        self.version
    }

    /// Checks the API version of the running proc-macro server.
    fn version_check(&self) -> Result<u32, ServerError> {
        let request = Request::ApiVersionCheck {};
        let response = self.send_task(request)?;

        match response {
            Response::ApiVersionCheck(version) => Ok(version),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    /// Enable support for rust-analyzer span mode if the server supports it.
    fn enable_rust_analyzer_spans(&self) -> Result<SpanMode, ServerError> {
        let request = Request::SetConfig(ServerConfig {
            span_mode: SpanMode::RustAnalyzer {
                fixup_ast_id: FIXUP_ERASED_FILE_AST_ID_MARKER.into_raw(),
            },
        });
        let response = self.send_task(request)?;

        match response {
            Response::SetConfig(ServerConfig { span_mode }) => Ok(span_mode),
            _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
        }
    }

    /// Finds proc-macros in a given dynamic library.
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

    /// Sends a request to the proc-macro server and waits for a response.
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

/// Manages the execution of the proc-macro server process.
#[derive(Debug)]
struct Process {
    child: JodChild,
}

impl Process {
    /// Runs a new proc-macro server process with the specified environment variables.
    fn run<'a>(
        path: &AbsPath,
        env: impl IntoIterator<
            Item = (impl AsRef<std::ffi::OsStr>, &'a Option<impl 'a + AsRef<std::ffi::OsStr>>),
        >,
    ) -> io::Result<Process> {
        let child = JodChild(mk_child(path, env)?);
        Ok(Process { child })
    }

    /// Retrieves stdin and stdout handles for the process.
    fn stdio(&mut self) -> Option<(ChildStdin, BufReader<ChildStdout>)> {
        let stdin = self.child.stdin.take()?;
        let stdout = self.child.stdout.take()?;
        let read = BufReader::new(stdout);

        Some((stdin, read))
    }
}

/// Creates and configures a new child process for the proc-macro server.
fn mk_child<'a>(
    path: &AbsPath,
    extra_env: impl IntoIterator<
        Item = (impl AsRef<std::ffi::OsStr>, &'a Option<impl 'a + AsRef<std::ffi::OsStr>>),
    >,
) -> io::Result<Child> {
    #[allow(clippy::disallowed_methods)]
    let mut cmd = Command::new(path);
    for env in extra_env {
        match env {
            (key, Some(val)) => cmd.env(key, val),
            (key, None) => cmd.env_remove(key),
        };
    }
    cmd.env("RUST_ANALYZER_INTERNALS_DO_NOT_USE", "this is unstable")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    if cfg!(windows) {
        let mut path_var = std::ffi::OsString::new();
        path_var.push(path.parent().unwrap().parent().unwrap());
        path_var.push("\\bin;");
        path_var.push(std::env::var_os("PATH").unwrap_or_default());
        cmd.env("PATH", path_var);
    }
    cmd.spawn()
}

/// Sends a request to the server and reads the response.
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
