// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A thin wrapper around `Command` in the standard library which allows us to
//! read the arguments that are built up.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io;
use std::process::{self, Output, Child};

pub struct Command {
    program: OsString,
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
}

impl Command {
    pub fn new<P: AsRef<OsStr>>(program: P) -> Command {
        Command::_new(program.as_ref())
    }

    fn _new(program: &OsStr) -> Command {
        Command {
            program: program.to_owned(),
            args: Vec::new(),
            env: Vec::new(),
        }
    }

    pub fn arg<P: AsRef<OsStr>>(&mut self, arg: P) -> &mut Command {
        self._arg(arg.as_ref());
        self
    }

    pub fn args<I>(&mut self, args: I) -> &mut Command
        where I: IntoIterator,
              I::Item: AsRef<OsStr>,
    {
        for arg in args {
            self._arg(arg.as_ref());
        }
        self
    }

    fn _arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_owned());
    }

    pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Command
        where K: AsRef<OsStr>,
              V: AsRef<OsStr>
    {
        self._env(key.as_ref(), value.as_ref());
        self
    }

    pub fn envs<I, K, V>(&mut self, envs: I) -> &mut Command
        where I: IntoIterator<Item=(K, V)>,
              K: AsRef<OsStr>,
              V: AsRef<OsStr>
    {
        for (key, value) in envs {
            self._env(key.as_ref(), value.as_ref());
        }
        self
    }

    fn _env(&mut self, key: &OsStr, value: &OsStr) {
        self.env.push((key.to_owned(), value.to_owned()));
    }

    pub fn output(&mut self) -> io::Result<Output> {
        self.command().output()
    }

    pub fn spawn(&mut self) -> io::Result<Child> {
        self.command().spawn()
    }

    pub fn command(&self) -> process::Command {
        let mut ret = process::Command::new(&self.program);
        ret.args(&self.args);
        ret.envs(self.env.clone());
        return ret
    }

    // extensions

    pub fn get_program(&self) -> &OsStr {
        &self.program
    }

    pub fn get_args(&self) -> &[OsString] {
        &self.args
    }

    pub fn get_env(&self) -> &[(OsString, OsString)] {
        &self.env
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.command().fmt(f)
    }
}
