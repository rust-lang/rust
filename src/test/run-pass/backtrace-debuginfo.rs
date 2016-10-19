// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We disable tail merging here because it can't preserve debuginfo and thus
// potentially breaks the backtraces. Also, subtle changes can decide whether
// tail merging suceeds, so the test might work today but fail tomorrow due to a
// seemingly completely unrelated change.
// Unfortunately, LLVM has no "disable" option for this, so we have to set
// "enable" to 0 instead.

// compile-flags:-g -Cllvm-args=-enable-tail-merge=0
// ignore-pretty issue #37195
// ignore-emscripten spawning processes is not supported

use std::io;
use std::io::prelude::*;
use std::env;

#[path = "backtrace-debuginfo-aux.rs"] mod aux;

macro_rules! pos {
    () => ((file!(), line!()))
}

macro_rules! dump_and_die {
    ($($pos:expr),*) => ({
        // FIXME(#18285): we cannot include the current position because
        // the macro span takes over the last frame's file/line.
        if cfg!(any(target_os = "macos",
                    target_os = "ios",
                    target_os = "android",
                    all(target_os = "linux", target_arch = "arm"),
                    target_os = "windows",
                    target_os = "freebsd",
                    target_os = "dragonfly",
                    target_os = "bitrig",
                    target_os = "openbsd")) {
            // skip these platforms as this support isn't implemented yet.
        } else {
            dump_filelines(&[$($pos),*]);
            panic!();
        }
    })
}

// we can't use a function as it will alter the backtrace
macro_rules! check {
    ($counter:expr; $($pos:expr),*) => ({
        if *$counter == 0 {
            dump_and_die!($($pos),*)
        } else {
            *$counter -= 1;
        }
    })
}

type Pos = (&'static str, u32);

// this goes to stdout and each line has to be occurred
// in the following backtrace to stderr with a correct order.
fn dump_filelines(filelines: &[Pos]) {
    // Skip top frame for MSVC, because it sees the macro rather than
    // the containing function.
    let skip = if cfg!(target_env = "msvc") {1} else {0};
    for &(file, line) in filelines.iter().rev().skip(skip) {
        // extract a basename
        let basename = file.split(&['/', '\\'][..]).last().unwrap();
        println!("{}:{}", basename, line);
    }
}

#[inline(never)]
fn inner(counter: &mut i32, main_pos: Pos, outer_pos: Pos) {
    check!(counter; main_pos, outer_pos);
    check!(counter; main_pos, outer_pos);
    let inner_pos = pos!(); aux::callback(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); aux::callback_inlined(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
}

// LLVM does not yet output the required debug info to support showing inlined
// function calls in backtraces when targetting MSVC, so disable inlining in
// this case.
#[cfg_attr(not(target_env = "msvc"), inline(always))]
#[cfg_attr(target_env = "msvc", inline(never))]
fn inner_inlined(counter: &mut i32, main_pos: Pos, outer_pos: Pos) {
    check!(counter; main_pos, outer_pos);
    check!(counter; main_pos, outer_pos);

    // Again, disable inlining for MSVC.
    #[cfg_attr(not(target_env = "msvc"), inline(always))]
    #[cfg_attr(target_env = "msvc", inline(never))]
    fn inner_further_inlined(counter: &mut i32, main_pos: Pos, outer_pos: Pos, inner_pos: Pos) {
        check!(counter; main_pos, outer_pos, inner_pos);
    }
    inner_further_inlined(counter, main_pos, outer_pos, pos!());

    let inner_pos = pos!(); aux::callback(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });
    let inner_pos = pos!(); aux::callback_inlined(|aux_pos| {
        check!(counter; main_pos, outer_pos, inner_pos, aux_pos);
    });

    // this tests a distinction between two independent calls to the inlined function.
    // (un)fortunately, LLVM somehow merges two consecutive such calls into one node.
    inner_further_inlined(counter, main_pos, outer_pos, pos!());
}

#[inline(never)]
fn outer(mut counter: i32, main_pos: Pos) {
    inner(&mut counter, main_pos, pos!());
    inner_inlined(&mut counter, main_pos, pos!());
}

fn check_trace(output: &str, error: &str) {
    // reverse the position list so we can start with the last item (which was the first line)
    let mut remaining: Vec<&str> = output.lines().map(|s| s.trim()).rev().collect();

    assert!(error.contains("stack backtrace"), "no backtrace in the error: {}", error);
    for line in error.lines() {
        if !remaining.is_empty() && line.contains(remaining.last().unwrap()) {
            remaining.pop();
        }
    }
    assert!(remaining.is_empty(),
            "trace does not match position list: {}\n---\n{}", error, output);
}

fn run_test(me: &str) {
    use std::str;
    use std::process::Command;

    let mut template = Command::new(me);
    template.env("RUST_BACKTRACE", "1");

    let mut i = 0;
    loop {
        let out = Command::new(me)
                          .env("RUST_BACKTRACE", "1")
                          .arg(i.to_string()).output().unwrap();
        let output = str::from_utf8(&out.stdout).unwrap();
        let error = str::from_utf8(&out.stderr).unwrap();
        if out.status.success() {
            assert!(output.contains("done."), "bad output for successful run: {}", output);
            break;
        } else {
            check_trace(output, error);
        }
        i += 1;
    }
}

#[inline(never)]
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 {
        let case = args[1].parse().unwrap();
        writeln!(&mut io::stderr(), "test case {}", case).unwrap();
        outer(case, pos!());
        println!("done.");
    } else {
        run_test(&args[0]);
    }
}

