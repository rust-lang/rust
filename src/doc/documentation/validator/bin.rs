// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "validator"]
#![crate_type = "bin"]
#![feature(phase)]
#![allow(unused_must_use)]

extern crate core;
extern crate regex;
extern crate native;
#[phase(syntax)] extern crate regex_macros;

use std::io::{TempDir, IoResult};
use std::io::fs::File;
use std::path::posix::Path;
use std::io::process::{Process,ProcessConfig,ProcessOutput,InheritFd,CreatePipe};
use regex::{Regex,Captures};
use std::os;
use std::io::fs;

pub struct Compiler {
    input: ~str,
    args: Vec<~str>,
    process: Option<Process>
}

impl Compiler {
    pub fn new(input: ~str) -> Compiler {
        Compiler {
            input: input,
            args: vec![],
            process: None
        }
    }

    pub fn config<'a>(&'a self) -> ProcessConfig<'a> {
        let mut config = ProcessConfig::new();

        config.program = "rustc";
        config.args = self.args.as_slice();

        config
    }


    pub fn exec(&mut self) -> Result<(), ~str>{
        // Create a tmp directory to stash all the files.
        let tmp = TempDir::new("exec_compiler").unwrap();

        // Save the path to that directory.
        let path = tmp.path();

        // Join a path for the block.rs file.
        let block_path = path.join("block.rs");

        // Create a new file with the previous path.
        let mut file = File::create(&block_path).unwrap();

        match file.write(self.input.as_bytes()) {
            Ok(r) => {},
            Err(err) => fail!("Oops: {}", err)
        }

        let f = File::open(&block_path);

        self.args.push(block_path.as_str().unwrap().to_owned());
        self.args.push("--out-dir".to_owned());
        self.args.push(path.as_str().unwrap().to_owned());

        let mut config = self.config();

        config.stdout = InheritFd(1);
        config.stderr = InheritFd(2);

        let mut process = Process::configure(config).unwrap();
        let exit = process.wait().unwrap();

        if exit.success() {
            Ok(())
        } else {
            let msg = format!("Could not execute process `{}`", exit);
            Err(msg)
        }
    }
}

pub struct Block<'a> {
    file: &'a Path,
    input: &'a str,
    start: uint,
    end: uint
}

impl<'a> Block<'a> {
    pub fn new(file: &'a Path, input: &'a str, start: uint, end: uint) -> Block<'a> {
        Block {
            file: file,
            input: input,
            start: start,
            end: end
        }
    }

    pub fn compile(&self) -> Result<(), ~str> {
        let mut compiler = Compiler::new(self.input.to_owned());
        compiler.exec()
    }
}

pub struct Page<'a, 'r, 't> {
    input: ~str,
    path: &'a Path,
    blocks: Vec<Block<'a>>
}

impl<'a, 'r, 't> Page<'a, 'r, 't> {
    pub fn new(path: &'a Path) -> Page<'a, 'r, 't> {
        let mut file = File::open(path).unwrap();

        Page {
            input: file.read_to_str().unwrap(),
            path: path,
            blocks: Vec::new()
        }
    }

    pub fn compile(&'t self, regex: &'r Regex) -> Result<uint, ~str> {
        let mut iter = regex.captures_iter(self.input);
        let mut count = 0;

        for capture in iter {
            count = count + 1;
            let (start, end) = capture.pos(1).unwrap();
            let block = Block::new(self.path, capture.at(1), start, end);
            try!(block.compile());
        }

        Ok(count)
    }
}

fn main() {
    // Skip a line to not crowd things.
    println!("");

    let mut args  = os::args();
    let dir       = args.pop().unwrap();
    let dir_path  = Path::new(StrBuf::from_owned_str(dir));

    let re = regex!(r"``` \{\.rust\}\n([^`]+)\n");

    let mut files = fs::walk_dir(&dir_path).unwrap();

    for file in files {
        let page = Page::new(&file);
        match page.compile(&re) {
            Ok(count) => println!(
                "Successfully compiled {} blocks from: {}", count,
                file.display()
            ),
            Err(err) => fail!("{}", err)
        }
    }
}