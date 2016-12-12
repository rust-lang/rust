#![allow(dead_code)]

use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::PathBuf;

use gcc;
use tempdir::TempDir;

pub struct Test {
    pub td: TempDir,
    pub gcc: PathBuf,
    pub msvc: bool,
}

pub struct Execution {
    args: Vec<String>,
}

impl Test {
    pub fn new() -> Test {
        let mut gcc = PathBuf::from(env::current_exe().unwrap());
        gcc.pop();
        if gcc.ends_with("deps") {
            gcc.pop();
        }
        gcc.push(format!("gcc-shim{}", env::consts::EXE_SUFFIX));
        Test {
            td: TempDir::new("gcc-test").unwrap(),
            gcc: gcc,
            msvc: false,
        }
    }

    pub fn gnu() -> Test {
        let t = Test::new();
        t.shim("cc").shim("ar");
        return t
    }

    pub fn msvc() -> Test {
        let mut t = Test::new();
        t.shim("cl").shim("lib.exe");
        t.msvc = true;
        return t
    }

    pub fn shim(&self, name: &str) -> &Test {
        let fname = format!("{}{}", name, env::consts::EXE_SUFFIX);
        fs::hard_link(&self.gcc, self.td.path().join(&fname)).or_else(|_| {
            fs::copy(&self.gcc, self.td.path().join(&fname)).map(|_| ())
        }).unwrap();
        self
    }

    pub fn gcc(&self) -> gcc::Config {
        let mut cfg = gcc::Config::new();
        let mut path = env::split_paths(&env::var_os("PATH").unwrap())
                           .collect::<Vec<_>>();
        path.insert(0, self.td.path().to_owned());
        let target = if self.msvc {
            "x86_64-pc-windows-msvc"
        } else {
            "x86_64-unknown-linux-gnu"
        };

        cfg.target(target).host(target)
           .opt_level(2)
           .debug(false)
           .out_dir(self.td.path())
           .__set_env("PATH", env::join_paths(path).unwrap())
           .__set_env("GCCTEST_OUT_DIR", self.td.path());
        if self.msvc {
            cfg.compiler(self.td.path().join("cl"));
            cfg.archiver(self.td.path().join("lib.exe"));
        }
        return cfg
    }

    pub fn cmd(&self, i: u32) -> Execution {
        let mut s = String::new();
        File::open(self.td.path().join(format!("out{}", i))).unwrap()
             .read_to_string(&mut s).unwrap();
        Execution {
            args: s.lines().map(|s| s.to_string()).collect(),
        }
    }
}

impl Execution {
    pub fn must_have<P: AsRef<OsStr>>(&self, p: P) -> &Execution {
        if !self.has(p.as_ref()) {
            panic!("didn't find {:?} in {:?}", p.as_ref(), self.args);
        } else {
            self
        }
    }

    pub fn must_not_have<P: AsRef<OsStr>>(&self, p: P) -> &Execution {
        if self.has(p.as_ref()) {
            panic!("found {:?}", p.as_ref());
        } else {
            self
        }
    }

    pub fn has(&self, p: &OsStr) -> bool {
        self.args.iter().any(|arg| {
            OsStr::new(arg) == p
        })
    }
}
