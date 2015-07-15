// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Run the command in a background thread. Redirect the output (and error
//! output) to the screen.

use std::process::{Stdio, Command};
use std::sync::mpsc::{channel, Receiver};
use std::thread;
use std::io::{self, Read, BufRead, BufReader, Write};

/// Run the command in another thread. Print output to screen.
/// Panics if command failed to run.
pub trait Run {
    fn run(&mut self);
}

#[derive(Clone, Copy)]
enum OutputType { Stdout, Stderr }

impl Run for Command {
    fn run(&mut self) {
        match run(self) {
            Ok(_) => {},
            Err(msg) => panic!(msg)
        }
    }
}

fn run(cmd : &mut Command) -> Result<(), String> {
    println!("Running {:?}", cmd);
    let mut child = try!(cmd.stdin(Stdio::piped())
                         .stdout(Stdio::piped())
                         .stderr(Stdio::piped())
                         .spawn()
                         .map_err(|e| format!("Failed: {:?}: {}", cmd, e)));
    let stdout = child.stdout.take().expect("log.rs: child.stdout");
    let stderr = child.stderr.take().expect("log.rs: child.stderr");
    let stdout_ch = read_async(stdout, OutputType::Stdout);
    let stderr_ch = read_async(stderr, OutputType::Stderr);
    let status = child.wait().expect("log.rs: child.wait");
    let _ = stdout_ch.recv();
    let _ = stderr_ch.recv();
    if status.success() {
        Ok(())
    } else {
        Err(format!("Command {:?} did not return a success status.", cmd))
    }
}

fn read_async<R : Read + Send + 'static>(reader : R, ty : OutputType)
                                         -> Receiver<()> {
    let (tx, rx) = channel();
    thread::spawn(move || {
        tx.send(log(reader, ty)).expect("log.rs: run_async()");
    });
    rx
}

fn log<R : Read + Send + 'static>(reader : R, ty : OutputType) {
    let mut buf_reader = BufReader::new(reader);
    let mut line : Vec<u8> = Vec::new();
    while buf_reader.read_until(0xA, &mut line).expect("log.rs: log()") > 0 {
        let _ = match ty {
            OutputType::Stdout => io::stdout().write_all(&line),
            OutputType::Stderr => io::stderr().write_all(&line)
        };
        line.clear();
    }
}
