// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi::{OsStr, OsString};
use std::io::Result;
use std::process;
use std::path::Path;

// std::process::Command but with addition support for add arguments past
// the `--` when invoking `cargo rustc`
#[derive(Debug)]
pub struct Command {
    cmd: process::Command,
    deferred_args: Vec<OsString>,
}

impl Command {

    pub fn new<S: AsRef<OsStr>>(program: S) -> Command {
        Command {
            cmd: process::Command::new(program),
            deferred_args: vec![],
        }
    }

    pub fn from_std_command(cmd: process::Command) -> Command {
        Command {
            cmd,
            deferred_args: vec![],
        }
    }

    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.cmd.arg(arg);
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Command
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.cmd.args(args);
        self
    }

    pub fn deferred_arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.deferred_args.push(arg.as_ref().to_owned());
        self
    }

    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.cmd.env(key, val);
        self
    }

    pub fn envs<I, K, V>(&mut self, vars: I) -> &mut Command
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.cmd.envs(vars);
        self
    }

    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        self.cmd.env_remove(key);
        self
    }

    pub fn env_clear(&mut self) -> &mut Command {
        self.cmd.env_clear();
        self
    }

    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Command {
        self.cmd.current_dir(dir);
        self
    }

    pub fn stdin<T: Into<process::Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.cmd.stdin(cfg);
        self
    }

    pub fn stdout<T: Into<process::Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.cmd.stdout(cfg);
        self
    }

    pub fn stderr<T: Into<process::Stdio>>(&mut self, cfg: T) -> &mut Command {
        self.cmd.stderr(cfg);
        self
    }

    pub fn spawn(&mut self) -> Result<process::Child> {
        self.cmd.args(self.deferred_args.clone().into_iter());
        self.cmd.spawn()
    }

    pub fn output(&mut self) -> Result<process::Output> {
        self.cmd.args(self.deferred_args.clone().into_iter());
        self.cmd.output()
    }

    pub fn status(&mut self) -> Result<process::ExitStatus> {
        self.cmd.args(self.deferred_args.clone().into_iter());
        self.cmd.status()
    }
}
