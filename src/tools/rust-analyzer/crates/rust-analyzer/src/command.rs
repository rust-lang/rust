//! Utilities for running a cargo command like `cargo check` or `cargo test` in a separate thread
//! and parse its stdout/stderr.

use std::{
    ffi::OsString,
    fmt, io,
    marker::PhantomData,
    path::PathBuf,
    process::{ChildStderr, ChildStdout, Command, Stdio},
};

use crossbeam_channel::Sender;
use process_wrap::std::{StdChildWrapper, StdCommandWrap};
use stdx::process::streaming_output;

/// Cargo output is structured as one JSON per line. This trait abstracts parsing one line of
/// cargo output into a Rust data type
pub(crate) trait CargoParser<T>: Send + 'static {
    fn from_line(&self, line: &str, error: &mut String) -> Option<T>;
    fn from_eof(&self) -> Option<T>;
}

struct CargoActor<T> {
    parser: Box<dyn CargoParser<T>>,
    sender: Sender<T>,
    stdout: ChildStdout,
    stderr: ChildStderr,
}

impl<T: Sized + Send + 'static> CargoActor<T> {
    fn new(
        parser: impl CargoParser<T>,
        sender: Sender<T>,
        stdout: ChildStdout,
        stderr: ChildStderr,
    ) -> Self {
        let parser = Box::new(parser);
        CargoActor { parser, sender, stdout, stderr }
    }
}

impl<T: Sized + Send + 'static> CargoActor<T> {
    fn run(self) -> io::Result<(bool, String)> {
        // We manually read a line at a time, instead of using serde's
        // stream deserializers, because the deserializer cannot recover
        // from an error, resulting in it getting stuck, because we try to
        // be resilient against failures.
        //
        // Because cargo only outputs one JSON object per line, we can
        // simply skip a line if it doesn't parse, which just ignores any
        // erroneous output.

        let mut stdout_errors = String::new();
        let mut stderr_errors = String::new();
        let mut read_at_least_one_stdout_message = false;
        let mut read_at_least_one_stderr_message = false;
        let process_line = |line: &str, error: &mut String| {
            // Try to deserialize a message from Cargo or Rustc.
            if let Some(t) = self.parser.from_line(line, error) {
                self.sender.send(t).unwrap();
                true
            } else {
                false
            }
        };
        let output = streaming_output(
            self.stdout,
            self.stderr,
            &mut |line| {
                if process_line(line, &mut stdout_errors) {
                    read_at_least_one_stdout_message = true;
                }
            },
            &mut |line| {
                if process_line(line, &mut stderr_errors) {
                    read_at_least_one_stderr_message = true;
                }
            },
            &mut || {
                if let Some(t) = self.parser.from_eof() {
                    self.sender.send(t).unwrap();
                }
            },
        );

        let read_at_least_one_message =
            read_at_least_one_stdout_message || read_at_least_one_stderr_message;
        let mut error = stdout_errors;
        error.push_str(&stderr_errors);
        match output {
            Ok(_) => Ok((read_at_least_one_message, error)),
            Err(e) => Err(io::Error::new(e.kind(), format!("{e:?}: {error}"))),
        }
    }
}

struct JodGroupChild(Box<dyn StdChildWrapper>);

impl Drop for JodGroupChild {
    fn drop(&mut self) {
        _ = self.0.kill();
        _ = self.0.wait();
    }
}

/// A handle to a cargo process used for fly-checking.
pub(crate) struct CommandHandle<T> {
    /// The handle to the actual cargo process. As we cannot cancel directly from with
    /// a read syscall dropping and therefore terminating the process is our best option.
    child: JodGroupChild,
    thread: stdx::thread::JoinHandle<io::Result<(bool, String)>>,
    program: OsString,
    arguments: Vec<OsString>,
    current_dir: Option<PathBuf>,
    _phantom: PhantomData<T>,
}

impl<T> fmt::Debug for CommandHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandHandle")
            .field("program", &self.program)
            .field("arguments", &self.arguments)
            .field("current_dir", &self.current_dir)
            .finish()
    }
}

impl<T: Sized + Send + 'static> CommandHandle<T> {
    pub(crate) fn spawn(
        mut command: Command,
        parser: impl CargoParser<T>,
        sender: Sender<T>,
    ) -> std::io::Result<Self> {
        command.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());

        let program = command.get_program().into();
        let arguments = command.get_args().map(|arg| arg.into()).collect::<Vec<OsString>>();
        let current_dir = command.get_current_dir().map(|arg| arg.to_path_buf());

        let mut child = StdCommandWrap::from(command);
        #[cfg(unix)]
        child.wrap(process_wrap::std::ProcessSession);
        #[cfg(windows)]
        child.wrap(process_wrap::std::JobObject);
        let mut child = child.spawn().map(JodGroupChild)?;

        let stdout = child.0.stdout().take().unwrap();
        let stderr = child.0.stderr().take().unwrap();

        let actor = CargoActor::<T>::new(parser, sender, stdout, stderr);
        let thread =
            stdx::thread::Builder::new(stdx::thread::ThreadIntent::Worker, "CommandHandle")
                .spawn(move || actor.run())
                .expect("failed to spawn thread");
        Ok(CommandHandle { program, arguments, current_dir, child, thread, _phantom: PhantomData })
    }

    pub(crate) fn cancel(mut self) {
        let _ = self.child.0.kill();
        let _ = self.child.0.wait();
    }

    pub(crate) fn join(mut self) -> io::Result<()> {
        let exit_status = self.child.0.wait()?;
        let (read_at_least_one_message, error) = self.thread.join()?;
        if read_at_least_one_message || exit_status.success() {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "Cargo watcher failed, the command produced no valid metadata (exit code: {exit_status:?}):\n{error}"
            )))
        }
    }
}
