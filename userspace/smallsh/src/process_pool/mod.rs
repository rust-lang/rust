#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub struct ProcessPool {
    processes: Vec<Child>,
    last_exit_code: Option<i32>,
    foreground_only: bool,
}

impl ProcessPool {
    pub fn new() -> Self {
        Self {
            processes: Vec::new(),
            last_exit_code: None,
            foreground_only: false,
        }
    }

    pub fn add(&mut self, cwd: &Path, program: &str, args: Vec<String>) -> io::Result<()> {
        let backgrounded = matches!(args.last().map(String::as_str), Some("&"));
        let allow_background = backgrounded && !self.foreground_only;
        let args_len = if backgrounded {
            args.len().saturating_sub(1)
        } else {
            args.len()
        };
        let parsed = ParsedCommand::from_args(args[..args_len].to_vec())?;
        let resolved_program = resolve_program(cwd, program);

        let mut command = Command::new(&resolved_program);
        command.args(&parsed.args).current_dir(cwd);

        if let Some(input) = parsed.input {
            command.stdin(File::open(resolve_io_path(cwd, &input))?);
        } else if allow_background {
            command.stdin(Stdio::null());
        }

        if let Some(output) = parsed.output {
            command.stdout(File::create(resolve_io_path(cwd, &output))?);
        } else if allow_background {
            command.stdout(Stdio::null());
        }

        if allow_background {
            let child = command.spawn()?;
            println!("Background process spawned with id: {}", child.id());
            self.processes.push(child);
        } else {
            let mut child = command.spawn()?;
            let status = child.wait()?;
            self.last_exit_code = Some(status.code().unwrap_or(1));
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.processes.len()
    }

    pub fn last_exit_code(&self) -> Option<i32> {
        self.last_exit_code
    }

    pub fn foreground_only(&self) -> bool {
        self.foreground_only
    }

    pub fn set_background(&mut self) {
        self.foreground_only = false;
    }

    pub fn set_foreground(&mut self) {
        self.foreground_only = true;
    }

    pub fn reap_finished(&mut self) {
        let mut idx = 0;
        while idx < self.processes.len() {
            match self.processes[idx].try_wait() {
                Ok(Some(status)) => {
                    let pid = self.processes[idx].id();
                    let code = status.code().unwrap_or(1);
                    self.last_exit_code = Some(code);
                    println!("Background process {} finished with code {}", pid, code);
                    self.processes.remove(idx);
                }
                Ok(None) => {
                    idx += 1;
                }
                Err(err) => {
                    let pid = self.processes[idx].id();
                    eprintln!("Failed to reap background process {}: {}", pid, err);
                    self.processes.remove(idx);
                }
            }
        }
    }

    pub fn terminate_all(&mut self) {
        for process in &mut self.processes {
            let _ = process.kill();
            let _ = process.wait();
        }
        self.processes.clear();
    }
}

struct ParsedCommand {
    args: Vec<String>,
    input: Option<String>,
    output: Option<String>,
}

impl ParsedCommand {
    fn from_args(args: Vec<String>) -> io::Result<Self> {
        let mut cleaned = Vec::new();
        let mut input = None;
        let mut output = None;
        let mut iter = args.into_iter();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "<" => {
                    input = Some(iter.next().ok_or_else(missing_redirection_target)?);
                }
                ">" => {
                    output = Some(iter.next().ok_or_else(missing_redirection_target)?);
                }
                _ => cleaned.push(arg),
            }
        }

        Ok(Self {
            args: cleaned,
            input,
            output,
        })
    }
}

fn missing_redirection_target() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "missing redirection target")
}

fn resolve_program(cwd: &Path, program: &str) -> PathBuf {
    let path = Path::new(program);
    if path.is_absolute() {
        path.to_path_buf()
    } else if program.contains('/') {
        cwd.join(path)
    } else {
        PathBuf::from("/").join(program)
    }
}

fn resolve_io_path(cwd: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    }
}
