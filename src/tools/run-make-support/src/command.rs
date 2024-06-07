use std::ffi::OsStr;
use std::io::Write;
use std::ops::{Deref, DerefMut};
use std::process::{Command as StdCommand, ExitStatus, Output, Stdio};

#[derive(Debug)]
pub struct Command {
    cmd: StdCommand,
    stdin: Option<Box<[u8]>>,
}

impl Command {
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        Self {
            cmd: StdCommand::new(program),
            stdin: None,
        }
    }

    pub fn set_stdin(&mut self, stdin: Box<[u8]>) {
        self.stdin = Some(stdin);
    }

    #[track_caller]
    pub(crate) fn command_output(&mut self) -> CompletedProcess {
        // let's make sure we piped all the input and outputs
        self.cmd.stdin(Stdio::piped());
        self.cmd.stdout(Stdio::piped());
        self.cmd.stderr(Stdio::piped());

        let output = if let Some(input) = &self.stdin {
            let mut child = self.cmd.spawn().unwrap();

            {
                let mut stdin = child.stdin.take().unwrap();
                stdin.write_all(input.as_ref()).unwrap();
            }

            child.wait_with_output().expect("failed to get output of finished process")
        } else {
            self.cmd.output().expect("failed to get output of finished process")
        };
        output.into()
    }
}

impl Deref for Command {
    type Target = StdCommand;

    fn deref(&self) -> &Self::Target {
        &self.cmd
    }
}

impl DerefMut for Command {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.cmd
    }
}

/// Represents the result of an executed process.
pub struct CompletedProcess {
    output: Output,
}

impl CompletedProcess {
    pub fn stdout_utf8(&self) -> String {
        String::from_utf8(self.output.stdout.clone()).expect("stdout is not valid UTF-8")
    }

    pub fn stderr_utf8(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("stderr is not valid UTF-8")
    }

    pub fn status(&self) -> ExitStatus {
        self.output.status
    }

    #[track_caller]
    pub fn assert_exit_code(&self, code: i32) {
        assert!(self.output.status.code() == Some(code));
    }
}

impl From<Output> for CompletedProcess {
    fn from(output: Output) -> Self {
        Self {
            output
        }
    }
}
