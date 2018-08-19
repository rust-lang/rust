// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is meant to be a comprehensive test of invocations with/without
// trailing commas (or other, similar optionally-trailing separators).
// Every macro is accounted for, even those not tested in this file.
// (There will be a note indicating why).

// std and core are both tested because they may contain separate
// implementations for some macro_rules! macros as an implementation
// detail.

// ignore-pretty issue #37195

// compile-flags: --test -C debug_assertions=yes
// revisions: std core

#![cfg_attr(core, no_std)]

#![feature(concat_idents)]

#[cfg(std)] use std::fmt;
#[cfg(core)] use core::fmt;

#[test]
fn assert() {
    assert!(true);
    assert!(true,);
    assert!(true, "hello");
    assert!(true, "hello",);
    assert!(true, "hello {}", "world");
    assert!(true, "hello {}", "world",);
}

#[test]
fn assert_eq() {
    assert_eq!(1, 1);
    assert_eq!(1, 1,);
    assert_eq!(1, 1, "hello");
    assert_eq!(1, 1, "hello",);
    assert_eq!(1, 1, "hello {}", "world");
    assert_eq!(1, 1, "hello {}", "world",);
}

#[test]
fn assert_ne() {
    assert_ne!(1, 2);
    assert_ne!(1, 2,);
    assert_ne!(1, 2, "hello");
    assert_ne!(1, 2, "hello",);
    assert_ne!(1, 2, "hello {}", "world");
    assert_ne!(1, 2, "hello {}", "world",);
}

#[test]
fn cfg() {
    cfg!(pants);
    cfg!(pants,);
    cfg!(pants = "pants");
    cfg!(pants = "pants",);
    cfg!(all(pants));
    cfg!(all(pants),);
    cfg!(all(pants,));
    cfg!(all(pants,),);
}

#[test]
fn column() {
    column!();
}

// compile_error! is in a companion to this test in compile-fail

#[test]
fn concat() {
    concat!();
    concat!("hello");
    concat!("hello",);
    concat!("hello", " world");
    concat!("hello", " world",);
}

#[test]
fn concat_idents() {
    fn foo() {}
    fn foobar() {}

    concat_idents!(foo)();
    concat_idents!(foo,)();
    concat_idents!(foo, bar)();
    concat_idents!(foo, bar,)();
}

#[test]
fn debug_assert() {
    debug_assert!(true);
    debug_assert!(true, );
    debug_assert!(true, "hello");
    debug_assert!(true, "hello",);
    debug_assert!(true, "hello {}", "world");
    debug_assert!(true, "hello {}", "world",);
}

#[test]
fn debug_assert_eq() {
    debug_assert_eq!(1, 1);
    debug_assert_eq!(1, 1,);
    debug_assert_eq!(1, 1, "hello");
    debug_assert_eq!(1, 1, "hello",);
    debug_assert_eq!(1, 1, "hello {}", "world");
    debug_assert_eq!(1, 1, "hello {}", "world",);
}

#[test]
fn debug_assert_ne() {
    debug_assert_ne!(1, 2);
    debug_assert_ne!(1, 2,);
    debug_assert_ne!(1, 2, "hello");
    debug_assert_ne!(1, 2, "hello",);
    debug_assert_ne!(1, 2, "hello {}", "world");
    debug_assert_ne!(1, 2, "hello {}", "world",);
}

#[test]
fn env() {
    env!("PATH");
    env!("PATH",);
    env!("PATH", "not found");
    env!("PATH", "not found",);
}

#[cfg(std)]
#[test]
fn eprint() {
    eprint!("hello");
    eprint!("hello",);
    eprint!("hello {}", "world");
    eprint!("hello {}", "world",);
}

#[cfg(std)]
#[test]
fn eprintln() {
    eprintln!();
    eprintln!("hello");
    eprintln!("hello",);
    eprintln!("hello {}", "world");
    eprintln!("hello {}", "world",);
}

#[test]
fn file() {
    file!();
}

#[cfg(std)]
#[test]
fn format() {
    format!("hello");
    format!("hello",);
    format!("hello {}", "world");
    format!("hello {}", "world",);
}

#[test]
fn format_args() {
    format_args!("hello");
    format_args!("hello",);
    format_args!("hello {}", "world");
    format_args!("hello {}", "world",);
}

#[test]
fn include() {
    include!("auxiliary/macro-comma-support.rs");
    include!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn include_bytes() {
    include_bytes!("auxiliary/macro-comma-support.rs");
    include_bytes!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn include_str() {
    include_str!("auxiliary/macro-comma-support.rs");
    include_str!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn line() {
    line!();
}

#[test]
fn module_path() {
    module_path!();
}

#[test]
fn option_env() {
    option_env!("PATH");
    option_env!("PATH",);
}

#[test]
fn panic() {
    // prevent 'unreachable code' warnings
    let falsum = || false;

    if falsum() { panic!(); }
    if falsum() { panic!("hello"); }
    if falsum() { panic!("hello",); }
    if falsum() { panic!("hello {}", "world"); }
    if falsum() { panic!("hello {}", "world",); }
}

#[cfg(std)]
#[test]
fn print() {
    print!("hello");
    print!("hello",);
    print!("hello {}", "world");
    print!("hello {}", "world",);
}

#[cfg(std)]
#[test]
fn println() {
    println!();
    println!("hello");
    println!("hello",);
    println!("hello {}", "world");
    println!("hello {}", "world",);
}

// select! is too troublesome and unlikely to be stabilized

// stringify! is N/A

#[cfg(std)]
#[test]
fn thread_local() {
    // this has an optional trailing *semicolon*
    thread_local! {
        #[allow(unused)] pub static A: () = ()
    }

    thread_local! {
        #[allow(unused)] pub static AA: () = ();
    }

    thread_local! {
        #[allow(unused)] pub static AAA: () = ();
        #[allow(unused)] pub static AAAA: () = ()
    }

    thread_local! {
        #[allow(unused)] pub static AAAAG: () = ();
        #[allow(unused)] pub static AAAAGH: () = ();
    }
}

#[test]
fn try() {
    fn inner() -> Result<(), ()> {
        try!(Ok(()));
        try!(Ok(()),);
        Ok(())
    }

    inner().unwrap();
}

#[test]
fn unimplemented() {
    // prevent 'unreachable code' warnings
    let falsum = || false;

    if falsum() { unimplemented!(); }
    if falsum() { unimplemented!("hello"); }
    if falsum() { unimplemented!("hello",); }
    if falsum() { unimplemented!("hello {}", "world"); }
    if falsum() { unimplemented!("hello {}", "world",); }
}

#[test]
fn unreachable() {
    // prevent 'unreachable code' warnings
    let falsum = || false;

    if falsum() { unreachable!(); }
    if falsum() { unreachable!("hello"); }
    if falsum() { unreachable!("hello",); }
    if falsum() { unreachable!("hello {}", "world"); }
    if falsum() { unreachable!("hello {}", "world",); }
}

#[cfg(std)]
#[test]
fn vec() {
    let _: Vec<()> = vec![];
    vec![0];
    vec![0,];
    vec![0, 1];
    vec![0, 1,];
}

// give a test body access to a fmt::Formatter, which seems
// to be the easiest way to use 'write!' on core.
macro_rules! test_with_formatter {
    (
        #[test]
        fn $fname:ident($f:ident: &mut fmt::Formatter) $block:block
    ) => {
        #[test]
        fn $fname() {
            struct Struct;
            impl fmt::Display for Struct {
                fn fmt(&self, $f: &mut fmt::Formatter) -> fmt::Result {
                    Ok($block)
                }
            }

            // suppress "unused"
            assert!(true, "{}", Struct);
        }
    };
}

test_with_formatter! {
    #[test]
    fn write(f: &mut fmt::Formatter) {
        write!(f, "hello");
        write!(f, "hello",);
        write!(f, "hello {}", "world");
        write!(f, "hello {}", "world",);
    }
}

test_with_formatter! {
    #[test]
    fn writeln(f: &mut fmt::Formatter) {
        writeln!(f);
        writeln!(f,);
        writeln!(f, "hello");
        writeln!(f, "hello",);
        writeln!(f, "hello {}", "world");
        writeln!(f, "hello {}", "world",);
    }
}
