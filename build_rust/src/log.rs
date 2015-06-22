// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module implements a simple logger that will pipe the output
//! of a `Command` to both the standard output and a log file.
//!
//! Both the stdout and stderr will be logged.

extern crate std;

use std::process::Command;
use std::process::Stdio;
use std::thread;
use std::path::{Path, PathBuf};
use std::io::{Read, BufRead, BufReader, Write};
use std::sync::mpsc::{channel, Receiver};
use build_state::*;

/// Log to <target-triple>/log/<prog>.{stdout,stderr}.log
pub struct Logger {
    stdout : PathBuf,
    stderr : PathBuf,
    triple : String,
    prog : String,
    verbose : bool
}

impl Logger {
    pub fn new(stdout : PathBuf, stderr : PathBuf,
           triple : String, prog : String, verbose : bool) -> Logger {
        Logger { stdout : stdout, stderr : stderr,
                 triple : triple, prog : prog, verbose : verbose }
    }
}

/// Tee pipes the output of a command to the stdio and a logger
pub trait Tee {
    fn tee(&mut self, logger : &Logger) -> BuildState<()>;
}

#[derive(Clone, Copy)]
enum OutputType {
    Stdout,
    Stderr
}

impl Tee for Command {
    fn tee(&mut self, logger : &Logger) -> BuildState<()> {
        let cmd = format!{"{:?}", self};
        if logger.verbose {
            println!("Running {}", cmd);
        }
        let mut child = try!(self
                             .stdin(Stdio::piped())
                             .stdout(Stdio::piped())
                             .stderr(Stdio::piped())
                             .spawn());
        let stdout =
            try!(child.stdout.take()
                 .ok_or(format!("{} failed: no stdout", logger.prog)));
        let stderr =
            try!(child.stderr.take()
                 .ok_or(format!("{} failed: no stderr", logger.prog)));
        let stdout_ch = read_async(&logger.stdout, &cmd, OutputType::Stdout,
                                   stdout, logger.verbose);
        let stderr_ch = read_async(&logger.stderr, &cmd, OutputType::Stderr,
                                   stderr, logger.verbose);
        let status = try!(child.wait());
        let _ = stdout_ch.recv(); // errors ignored
        let _ = stderr_ch.recv(); // errors ignored
        if !status.success() {
            err_stop!("{} failed. See log files under {}/log.",
                      logger.prog, logger.triple);
        }
        continue_build()
    }
}

fn read_async<R : Read + Send + 'static>(path : &Path,
                                         cmd : &str,
                                         out_type : OutputType,
                                         reader : R,
                                         verbose : bool)
                                         -> Receiver<BuildState<()>> {
    let top_line = match out_type {
        OutputType::Stdout => format!("# stdout {}\n", cmd),
        OutputType::Stderr => format!("# stderr {}\n", cmd)
    };
    let (tx, rx) = channel();
    let pathbuf = path.to_path_buf();
    thread::spawn(move || {
        tx.send(log_to_file(pathbuf, top_line, out_type,
                            reader, verbose)).unwrap();
    });
    rx
}

fn log_to_file<R : Read + Send + 'static>(path : PathBuf,
                                          top_line : String,
                                          out_type : OutputType,
                                          reader : R,
                                          verbose : bool)
                                          -> BuildState<()> {
    use std::io;
    use std::fs::OpenOptions;
    let mut buf_reader = BufReader::new(reader);
    let mut line : Vec<u8> = Vec::new();
    let mut file = try!(OpenOptions::new().create(true)
                        .write(true).append(true).open(&path)
                        .map_err(|e| format!(
                            "Failed to open file {:?}: {}", path, e)));
    try!(file.write_all(top_line.as_bytes()));
    while try!(buf_reader.read_until(0xA, &mut line)) > 0 {
        if verbose {
            try!(match out_type {
                OutputType::Stdout => io::stdout().write_all(&line),
                OutputType::Stderr => io::stderr().write_all(&line)
            });
        }
        try!(file.write_all(&line));
        line.clear();
    }
    continue_build()
}
