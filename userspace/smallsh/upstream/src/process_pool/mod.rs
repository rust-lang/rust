#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use std::fs::File;
use std::io::{self, Write};
use std::process::{id, Child, Command, Stdio};
use std::sync::{mpsc::TryRecvError, Arc, Mutex};

pub struct ProcessPool {
    processes: Arc<Mutex<Vec<Child>>>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    tx: std::sync::mpsc::Sender<()>,
    last_exit_code: Arc<Mutex<Option<i32>>>,
    foreground_only: bool,
}

impl Drop for ProcessPool {
    fn drop(&mut self) {
        self.tx
            .send(())
            .expect("Couldn't send closing signal to second thread.");
        self.thread_handle
            .take()
            .expect("This join handle was alread `take'd` somehow.")
            .join()
            .expect("Couldn't load process pool thread.");
    }
}

impl ProcessPool {
    pub fn new() -> Self {
        let processes = Arc::new(Mutex::new(vec![]));
        let code = Arc::new(Mutex::new(None));
        let (tx, rx) = std::sync::mpsc::channel();
        let cloned = processes.clone();
        let code_cloned = code.clone();

        ProcessPool {
            processes: processes,
            thread_handle: Some(std::thread::spawn(move || loop {
                let mut processes = cloned
                    .lock()
                    .expect("Couldn't lock processes for draining.");

                processes.drain_filter(|process| {
                    let finished = process.try_wait();

                    match finished {
                        Ok(Some(status)) => {
                            *code_cloned.lock().expect(
                                "Couldn't aquire last exit code mutex lock for setting code.",
                            ) = status.code();
                            true
                        }
                        _ => false,
                    }
                });

                match rx.try_recv() {
                    Ok(_) => break,
                    Err(TryRecvError::Disconnected) => break,
                    _ => {}
                }
            })),
            tx: tx,
            last_exit_code: code,
            foreground_only: false,
        }
    }

    fn extract_input_redirection(mut args: Vec<String>) -> (Option<File>, Vec<String>) {
        let iterator = args.iter().peekable();
        let mut file: Option<File> = None;

        let mut idx = -1;
        for (index, arg) in iterator.enumerate() {
            if arg == &"<" {
                idx = index as i32;
            }
        }

        if idx >= 0 && args.len() >= 2 {
            args.remove(idx as usize);

            file = match File::open(args.remove(idx as usize)) {
                Ok(file) => Some(file),
                _ => None,
            };
        }

        (file, args)
    }

    fn extract_output_redirection(mut args: Vec<String>) -> (Option<File>, Vec<String>) {
        let iterator = args.iter();
        let mut file: Option<File> = None;

        let mut idx = -1;
        for (index, arg) in iterator.enumerate() {
            if arg == &">" {
                idx = index as i32;
            }
        }

        if idx >= 0 && args.len() >= 2 {
            args.remove(idx as usize);
            file = match File::create(args.remove(idx as usize)) {
                Ok(file) => Some(file),
                _ => None,
            };        }

        (file, args)
    }

    fn replace_with_pid(args: Vec<&str>) -> Vec<String> {

        args.iter().map(|arg| {

            if arg == &"$$" {
                alloc::format!("{}", id())
            } else {
                arg.to_string()
            }

        }).collect()

    }

    pub fn add(&mut self, command: &str, mut args: Vec<&str>) -> io::Result<()> {
        let backgrounded = match args.last() {
            Some(&arg) if arg == "&" => {
                args.pop();
                true
            }
            _ => false,
        };

        let args = ProcessPool::replace_with_pid(args);
        let (input, args) = ProcessPool::extract_input_redirection(args);
        let (output, args) = ProcessPool::extract_output_redirection(args);

        let command = if backgrounded && !self.foreground_only {
            match (input, output) {
                (Some(input), Some(output)) => Some(
                    Command::new(command)
                        .args(args)
                        .stdin(input)
                        .stdout(output)
                        .spawn()?,
                ),
                (Some(input), _) => Some(Command::new(command).args(args).stdin(input).spawn()?),
                (_, Some(output)) => Some(
                    Command::new(command)
                        .args(args)
                        .stdin(Stdio::null())
                        .stdout(output)
                        .spawn()?,
                ),
                (_, _) => Some(
                    Command::new(command)
                        .args(args)
                        .stdin(Stdio::null())
                        .spawn()?,
                ),
            }
        } else {
            let mut command = match (input, output) {
                (Some(input), Some(output)) => Command::new(command)
                    .args(args)
                    .stdin(input)
                    .stdout(output)
                    .spawn()?,
                (Some(input), _) => Command::new(command).args(args).stdin(input).spawn()?,
                (_, Some(output)) => Command::new(command)
                    .args(args)
                    .stdin(Stdio::null())
                    .stdout(output)
                    .spawn()?,
                (_, _) => Command::new(command)
                    .args(args)
                    .stdin(Stdio::null())
                    .spawn()?,
            };

            match command.wait() {
                Ok(status) => {
                    *self
                        .last_exit_code
                        .lock()
                        .expect("Couldn't aquire lock for exit code after wait().") = status.code()
                }
                _ => eprintln!("Process failed to complete."),
            }

            None
        };

        if let Some(command) = command {
            println!("Background process spawned with id: {}", command.id());
            io::stdout().flush().ok().expect("Could not flush stdout");
            self.processes
                .lock()
                .expect("Couldn't aquire mutex lock for processes.")
                .push(command);
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.processes
            .lock()
            .expect("Couldn't aquire mutex lock for length.")
            .len()
    }

    pub fn last_exit_code(&self) -> Option<i32> {
        *self
            .last_exit_code
            .lock()
            .expect("Couldn't aquire mutex lock for exit code.")
    }

    pub fn foreground_only(&self) -> bool {
        self.foreground_only
    }

    pub fn set_foreground(&mut self) {
        self.foreground_only = true;
    }

    pub fn set_background(&mut self) {
        self.foreground_only = false;
    }
}
