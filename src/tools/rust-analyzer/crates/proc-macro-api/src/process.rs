//! Handle process life-time and message passing for proc-macro client

use std::{
    fmt::Debug,
    io::{self, BufRead, BufReader, Read, Write},
    panic::AssertUnwindSafe,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU32, Ordering},
    },
};

use paths::AbsPath;
use semver::Version;
use span::Span;
use stdx::JodChild;

use crate::{
    ProcMacro, ProcMacroKind, ProtocolFormat, ServerError,
    bidirectional_protocol::{self, SubCallback, msg::BidirectionalMessage, reject_subrequests},
    legacy_protocol::{self, SpanMode},
    version,
};

/// Represents a process handling proc-macro communication.
pub(crate) struct ProcMacroServerProcess {
    /// The state of the proc-macro server process, the protocol is currently strictly sequential
    /// hence the lock on the state.
    state: Mutex<ProcessSrvState>,
    version: u32,
    protocol: Protocol,
    /// Populated when the server exits.
    exited: OnceLock<AssertUnwindSafe<ServerError>>,
    active: AtomicU32,
}

impl std::fmt::Debug for ProcMacroServerProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcMacroServerProcess")
            .field("version", &self.version)
            .field("protocol", &self.protocol)
            .field("exited", &self.exited)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Protocol {
    LegacyJson { mode: SpanMode },
    BidirectionalPostcardPrototype { mode: SpanMode },
}

pub trait ProcessExit: Send + Sync {
    fn exit_err(&mut self) -> Option<ServerError>;
}

impl ProcessExit for Process {
    fn exit_err(&mut self) -> Option<ServerError> {
        match self.child.try_wait() {
            Ok(None) | Err(_) => None,
            Ok(Some(status)) => {
                let mut msg = String::new();
                if !status.success()
                    && let Some(stderr) = self.child.stderr.as_mut()
                {
                    _ = stderr.read_to_string(&mut msg);
                }
                Some(ServerError {
                    message: format!(
                        "proc-macro server exited with {status}{}{msg}",
                        if msg.is_empty() { "" } else { ": " }
                    ),
                    io: None,
                })
            }
        }
    }
}

/// Maintains the state of the proc-macro server process.
pub(crate) struct ProcessSrvState {
    process: Box<dyn ProcessExit>,
    stdin: Box<dyn Write + Send + Sync>,
    stdout: Box<dyn BufRead + Send + Sync>,
}

impl ProcMacroServerProcess {
    /// Starts the proc-macro server and performs a version check
    pub(crate) fn spawn<'a>(
        process_path: &AbsPath,
        env: impl IntoIterator<
            Item = (impl AsRef<std::ffi::OsStr>, &'a Option<impl 'a + AsRef<std::ffi::OsStr>>),
        > + Clone,
        version: Option<&Version>,
    ) -> io::Result<ProcMacroServerProcess> {
        Self::run(
            |format| {
                let mut process = Process::run(
                    process_path,
                    env.clone(),
                    format.map(|format| format.to_string()).as_deref(),
                )?;
                let (stdin, stdout) = process.stdio().expect("couldn't access child stdio");

                Ok((Box::new(process), Box::new(stdin), Box::new(stdout)))
            },
            version,
            || {
                #[expect(clippy::disallowed_methods)]
                Command::new(process_path)
                    .arg("--version")
                    .output()
                    .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                    .unwrap_or_else(|_| "unknown version".to_owned())
            },
        )
    }

    /// Invokes `spawn` and performs a version check.
    pub(crate) fn run(
        spawn: impl Fn(
            Option<ProtocolFormat>,
        ) -> io::Result<(
            Box<dyn ProcessExit>,
            Box<dyn Write + Send + Sync>,
            Box<dyn BufRead + Send + Sync>,
        )>,
        version: Option<&Version>,
        binary_server_version: impl Fn() -> String,
    ) -> io::Result<ProcMacroServerProcess> {
        const VERSION: Version = Version::new(1, 93, 0);
        // we do `>` for nightly as this started working in the middle of the 1.93 nightly release, so we dont want to break on half of the nightlies
        let has_working_format_flag = version.map_or(false, |v| {
            if v.pre.as_str() == "nightly" { *v > VERSION } else { *v >= VERSION }
        });

        let formats: &[_] = if std::env::var_os("RUST_ANALYZER_USE_POSTCARD").is_some()
            && has_working_format_flag
        {
            &[
                Some(ProtocolFormat::BidirectionalPostcardPrototype),
                Some(ProtocolFormat::JsonLegacy),
            ]
        } else {
            &[None]
        };

        let mut err = None;
        for &format in formats {
            let create_srv = || {
                let (process, stdin, stdout) = spawn(format)?;

                io::Result::Ok(ProcMacroServerProcess {
                    state: Mutex::new(ProcessSrvState { process, stdin, stdout }),
                    version: 0,
                    protocol: match format {
                        Some(ProtocolFormat::BidirectionalPostcardPrototype) => {
                            Protocol::BidirectionalPostcardPrototype { mode: SpanMode::Id }
                        }
                        Some(ProtocolFormat::JsonLegacy) | None => {
                            Protocol::LegacyJson { mode: SpanMode::Id }
                        }
                    },
                    exited: OnceLock::new(),
                    active: AtomicU32::new(0),
                })
            };
            let mut srv = create_srv()?;
            tracing::info!("sending proc-macro server version check");
            match srv.version_check(Some(&reject_subrequests)) {
                Ok(v) if v > version::CURRENT_API_VERSION => {
                    let process_version = binary_server_version();
                    err = Some(io::Error::other(format!(
                        "Your installed proc-macro server is too new for your rust-analyzer. API version: {}, server version: {process_version}. \
                        This will prevent proc-macro expansion from working. Please consider updating your rust-analyzer to ensure compatibility with your current toolchain.",
                        version::CURRENT_API_VERSION
                    )));
                }
                Ok(v) => {
                    tracing::info!("Proc-macro server version: {v}");
                    srv.version = v;
                    if srv.version >= version::RUST_ANALYZER_SPAN_SUPPORT
                        && let Ok(new_mode) =
                            srv.enable_rust_analyzer_spans(Some(&reject_subrequests))
                    {
                        match &mut srv.protocol {
                            Protocol::LegacyJson { mode }
                            | Protocol::BidirectionalPostcardPrototype { mode } => *mode = new_mode,
                        }
                    }
                    tracing::info!("Proc-macro server protocol: {:?}", srv.protocol);
                    return Ok(srv);
                }
                Err(e) => {
                    tracing::info!(%e, "proc-macro version check failed");
                    err = Some(io::Error::other(format!(
                        "proc-macro server version check failed: {e}"
                    )))
                }
            }
        }
        Err(err.unwrap())
    }

    /// Finds proc-macros in a given dynamic library.
    pub(crate) fn find_proc_macros(
        &self,
        dylib_path: &AbsPath,
        callback: Option<SubCallback<'_>>,
    ) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::find_proc_macros(self, dylib_path),

            Protocol::BidirectionalPostcardPrototype { .. } => {
                let cb = callback.expect("callback required for bidirectional protocol");
                bidirectional_protocol::find_proc_macros(self, dylib_path, cb)
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
            Protocol::BidirectionalPostcardPrototype { mode } => mode == SpanMode::RustAnalyzer,
        }
    }

    /// Checks the API version of the running proc-macro server.
    fn version_check(&self, callback: Option<SubCallback<'_>>) -> Result<u32, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::version_check(self),
            Protocol::BidirectionalPostcardPrototype { .. } => {
                let cb = callback.expect("callback required for bidirectional protocol");
                bidirectional_protocol::version_check(self, cb)
            }
        }
    }

    /// Enable support for rust-analyzer span mode if the server supports it.
    fn enable_rust_analyzer_spans(
        &self,
        callback: Option<SubCallback<'_>>,
    ) -> Result<SpanMode, ServerError> {
        match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::enable_rust_analyzer_spans(self),
            Protocol::BidirectionalPostcardPrototype { .. } => {
                let cb = callback.expect("callback required for bidirectional protocol");
                bidirectional_protocol::enable_rust_analyzer_spans(self, cb)
            }
        }
    }

    pub(crate) fn expand(
        &self,
        proc_macro: &ProcMacro,
        subtree: tt::SubtreeView<'_>,
        attr: Option<tt::SubtreeView<'_>>,
        env: Vec<(String, String)>,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
        callback: Option<SubCallback<'_>>,
    ) -> Result<Result<tt::TopSubtree, String>, ServerError> {
        self.active.fetch_add(1, Ordering::AcqRel);
        let result = match self.protocol {
            Protocol::LegacyJson { .. } => legacy_protocol::expand(
                proc_macro,
                self,
                subtree,
                attr,
                env,
                def_site,
                call_site,
                mixed_site,
                current_dir,
            ),
            Protocol::BidirectionalPostcardPrototype { .. } => bidirectional_protocol::expand(
                proc_macro,
                self,
                subtree,
                attr,
                env,
                def_site,
                call_site,
                mixed_site,
                current_dir,
                callback.expect("callback required for bidirectional protocol"),
            ),
        };

        self.active.fetch_sub(1, Ordering::AcqRel);
        result
    }

    pub(crate) fn send_task_legacy<Request, Response>(
        &self,
        send: impl FnOnce(
            &mut dyn Write,
            &mut dyn BufRead,
            Request,
            &mut String,
        ) -> Result<Option<Response>, ServerError>,
        req: Request,
    ) -> Result<Response, ServerError> {
        self.with_locked_io(String::new(), |writer, reader, buf| {
            send(writer, reader, req, buf).and_then(|res| {
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
        })
    }

    fn with_locked_io<R, B>(
        &self,
        mut buf: B,
        f: impl FnOnce(&mut dyn Write, &mut dyn BufRead, &mut B) -> Result<R, ServerError>,
    ) -> Result<R, ServerError> {
        let state = &mut *self.state.lock().unwrap();
        f(&mut state.stdin, &mut state.stdout, &mut buf).map_err(|e| {
            if e.io.as_ref().map(|it| it.kind()) == Some(io::ErrorKind::BrokenPipe) {
                match state.process.exit_err() {
                    None => e,
                    Some(server_error) => {
                        self.exited.get_or_init(|| AssertUnwindSafe(server_error)).0.clone()
                    }
                }
            } else {
                e
            }
        })
    }

    pub(crate) fn run_bidirectional(
        &self,
        initial: BidirectionalMessage,
        callback: SubCallback<'_>,
    ) -> Result<BidirectionalMessage, ServerError> {
        self.with_locked_io(Vec::new(), |writer, reader, buf| {
            bidirectional_protocol::run_conversation(writer, reader, buf, initial, callback)
        })
    }

    pub(crate) fn number_of_active_req(&self) -> u32 {
        self.active.load(Ordering::Acquire)
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
        format: Option<&str>,
    ) -> io::Result<Process> {
        let child = JodChild(mk_child(path, env, format)?);
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
    format: Option<&str>,
) -> io::Result<Child> {
    #[allow(clippy::disallowed_methods)]
    let mut cmd = Command::new(path);
    for env in extra_env {
        match env {
            (key, Some(val)) => cmd.env(key, val),
            (key, None) => cmd.env_remove(key),
        };
    }
    if let Some(format) = format {
        cmd.arg("--format");
        cmd.arg(format);
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
