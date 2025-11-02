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
    ProcMacroKind, ServerError,
    legacy_protocol::{self, SpanMode},
    version,
};

/// Represents a process handling proc-macro communication.
#[derive(Debug)]
pub(crate) struct ProcMacroServerProcess {
    /// The state of the proc-macro server process, the protocol is currently strictly sequential
    /// hence the lock on the state.
    state: Mutex<ProcessSrvState>,
    version: u32,
    protocol: Protocol,
    /// Populated when the server exits.
    exited: OnceLock<AssertUnwindSafe<ServerError>>,
}

#[derive(Debug)]
enum Protocol {
    LegacyJson { mode: SpanMode },
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
                protocol: Protocol::LegacyJson { mode: SpanMode::Id },
                exited: OnceLock::new(),
            })
        };
        let mut srv = create_srv()?;
        tracing::info!("sending proc-macro server version check");
        match srv.version_check() {
            Ok(v) if v > version::CURRENT_API_VERSION => {
                #[allow(clippy::disallowed_methods)]
                let process_version = Command::new(process_path)
                    .arg("--version")
                    .output()
                    .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                    .unwrap_or_else(|_| "unknown version".to_owned());
                Err(io::Error::other(format!(
                    "Your installed proc-macro server is too new for your rust-analyzer. API version: {}, server version: {process_version}. \
                        This will prevent proc-macro expansion from working. Please consider updating your rust-analyzer to ensure compatibility with your current toolchain.",
                    version::CURRENT_API_VERSION
                )))
            }
            Ok(v) => {
                tracing::info!("Proc-macro server version: {v}");
                srv.version = v;
                if srv.version >= version::RUST_ANALYZER_SPAN_SUPPORT
                    && let Ok(mode) = srv.enable_rust_analyzer_spans()
                {
                    srv.protocol = Protocol::LegacyJson { mode };
                }
                tracing::info!("Proc-macro server protocol: {:?}", srv.protocol);
                Ok(srv)
            }
            Err(e) => {
                tracing::info!(%e, "proc-macro version check failed");
                Err(io::Error::other(format!("proc-macro server version check failed: {e}")))
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

    /// Enable support for rust-analyzer span mode if the server supports it.
    pub(crate) fn rust_analyzer_spans(&self) -> bool {
        match self.protocol {
            Protocol::LegacyJson { mode } => mode == SpanMode::RustAnalyzer,
        }
    }

    /// Checks the API version of the running proc-macro server.
    fn version_check(&self) -> Result<u32, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::version_check(self),
        }
    }

    /// Enable support for rust-analyzer span mode if the server supports it.
    fn enable_rust_analyzer_spans(&self) -> Result<SpanMode, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::enable_rust_analyzer_spans(self),
        }
    }

    /// Finds proc-macros in a given dynamic library.
    pub(crate) fn find_proc_macros(
        &self,
        dylib_path: &AbsPath,
    ) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::find_proc_macros(self, dylib_path),
        }
    }

    pub(crate) fn send_task<Request, Response>(
        &self,
        serialize_req: impl FnOnce(
            &mut dyn Write,
            &mut dyn BufRead,
            Request,
            &mut String,
        ) -> Result<Option<Response>, ServerError>,
        req: Request,
    ) -> Result<Response, ServerError> {
        let state = &mut *self.state.lock().unwrap();
        let mut buf = String::new();
        serialize_req(&mut state.stdin, &mut state.stdout, req, &mut buf)
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
                            if !status.success()
                                && let Some(stderr) = state.process.child.stderr.as_mut()
                            {
                                _ = stderr.read_to_string(&mut msg);
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
