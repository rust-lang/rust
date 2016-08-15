// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows - this is a unix-specific test
// ignore-emscripten

#![feature(process_exec, libc)]

extern crate libc;

use std::env;
use std::io::Error;
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() {
    if let Some(arg) = env::args().skip(1).next() {
        match &arg[..] {
            "test1" => println!("hello2"),
            "test2" => assert_eq!(env::var("FOO").unwrap(), "BAR"),
            "test3" => assert_eq!(env::current_dir().unwrap()
                                      .to_str().unwrap(), "/"),
            "empty" => {}
            _ => panic!("unknown argument: {}", arg),
        }
        return
    }

    let me = env::current_exe().unwrap();

    let output = Command::new(&me).arg("test1").before_exec(|| {
        println!("hello");
        Ok(())
    }).output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout, b"hello\nhello2\n");

    let output = Command::new(&me).arg("test2").before_exec(|| {
        env::set_var("FOO", "BAR");
        Ok(())
    }).output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());

    let output = Command::new(&me).arg("test3").before_exec(|| {
        env::set_current_dir("/").unwrap();
        Ok(())
    }).output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());

    let output = Command::new(&me).arg("bad").before_exec(|| {
        Err(Error::from_raw_os_error(102))
    }).output().unwrap_err();
    assert_eq!(output.raw_os_error(), Some(102));

    let pid = unsafe { libc::getpid() };
    assert!(pid >= 0);
    let output = Command::new(&me).arg("empty").before_exec(move || {
        let child = unsafe { libc::getpid() };
        assert!(child >= 0);
        assert!(pid != child);
        Ok(())
    }).output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());

    let mem = Arc::new(AtomicUsize::new(0));
    let mem2 = mem.clone();
    let output = Command::new(&me).arg("empty").before_exec(move || {
        assert_eq!(mem2.fetch_add(1, Ordering::SeqCst), 0);
        Ok(())
    }).output().unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(output.stdout.is_empty());
    assert_eq!(mem.load(Ordering::SeqCst), 0);
}
