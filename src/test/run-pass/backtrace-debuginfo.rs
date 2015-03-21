// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-g
// ignore-pretty as this critically relies on line numbers

use std::io;
use std::io::prelude::*;
use std::env;

#[path = "backtrace-debuginfo-aux.rs"] mod aux;

macro_rules! pos {
    () => ((file!(), line!()))
}

#[cfg(all(unix,
          not(target_os = "macos"),
          not(target_os = "ios"),
          not(target_os = "android"),
          not(all(target_os = "linux", target_arch = "arm"))))]
macro_rules! dump_and_die {
    ($($pos:expr),*) => ({
        // FIXME(#18285): we cannot include the current position because
        // the macro span takes over the last frame's file/line.
        dump_filelines(&[$($pos),*]);
        panic!();
    })
}

// this does not work on Windows, Android, OSX or iOS
#[cfg(any(not(unix),
          target_os = "macos",
          target_os = "ios",
          target_os = "android",
          all(target_os = "linux", target_arch = "arm")))]
macro_rules! dump_and_die {
    ($($pos:expr),*) => ({ let _ = [$($pos),*]; })
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
    for &(file, line) in filelines.iter().rev() {
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

#[inline(always)]
fn inner_inlined(counter: &mut i32, main_pos: Pos, outer_pos: Pos) {
    check!(counter; main_pos, outer_pos);
    check!(counter; main_pos, outer_pos);

    #[inline(always)]
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
