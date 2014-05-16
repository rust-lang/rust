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
    dir: TempDir,
    args: Vec<~str>,
    process: Option<Process>
}

impl Compiler {
    pub fn new(input: ~str) -> Compiler {
        let tmp = TempDir::new("exec-compiler").unwrap();
        let path = tmp.path().as_str().unwrap().to_owned();

        Compiler {
            dir: tmp,
            args: vec![input, "--out-dir".to_owned(), path],
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
        let mut config = self.config();

        config.stdout = InheritFd(1);
        config.stderr = InheritFd(2);

        let mut process = Process::configure(config).unwrap();
        let exit = process.wait();

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
    start: uint,
    end: uint
}

impl<'a> Block<'a> {
    pub fn new(file: &'a Path, start: uint, end: uint) -> Block<'a> {
        Block {
            file: file,
            start: start,
            end: end
        }
    }

    pub fn compile(&self) -> Result<(), ~str> {
        let mut compiler = Compiler::new(self.file.as_str().unwrap().to_owned());
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

    pub fn compile(&'t self, regex: &'r Regex) -> Result<(), ~str> {
        let mut iter = regex.captures_iter(self.input);

        for capture in iter {
            let block = self.block(self.path, &capture);
            try!(block.compile());
        }

        Ok(())
    }

    pub fn block(&self, path: &'a Path, capture: &'t Captures<'t>) -> Block<'a> {
        let (start, end) = capture.pos(1).unwrap();
        Block::new(path, start, end)
    }
}

fn main() {
    println!("");

    // Create a new temporary directory that we can work within:
    let tmp_dir = TempDir::new("validator-block").unwrap();

    let mut args  = os::args();
    let dir       = args.pop().unwrap();
    let dir_path  = Path::new(StrBuf::from_owned_str(dir));

    // Walk the directory and capture all the
    // markdown files.

    // for path in dirs {
    //     let file = fs::File::open(&path).read_to_str().unwrap();
    //     let caps = re.captures(file.as_slice());
    //     println!("{}", caps.at(1));
    // }

    // Precompile the regular expression to match code blocks.
    let re = regex!(r"``` \{\.rust\}\n([^`]+)\n");

    let mut files = fs::walk_dir(&dir_path).unwrap();

    for file in files {
        let page = Page::new(&file);
        page.compile(&re);
    }

    let raw = r"
    ``` {.rust}
    fn main() {}
    ```";

    let mut i = 0;

    for caps in re.captures_iter(raw) {
        i = i + 1;
        let block = caps.at(1);
        let mut path  = Path::new("");

        path.push(tmp_dir.path());
        path.push(Path::new(format!("block-{}.rs", i)));

        // Create a new file
        let mut file = File::create(&path).unwrap();
        file.write(block.as_bytes());

        let (start, end) = caps.pos(1).unwrap();

        let mut block = Block::new(&path, start, end);
        match block.compile() {
            Ok(r) => println!("Success"),
            Err(err) => fail!("{}", err)
        }
    }
}