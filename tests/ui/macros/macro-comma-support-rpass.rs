//@ run-pass
// This is meant to be a comprehensive test of invocations with/without
// trailing commas (or other, similar optionally-trailing separators).
// Every macro is accounted for, even those not tested in this file.
// (There will be a note indicating why).

// std and core are both tested because they may contain separate
// implementations for some macro_rules! macros as an implementation
// detail.


//@ compile-flags: --test -C debug_assertions=yes
//@ revisions: std core

#![cfg_attr(core, no_std)]

#![allow(deprecated)] // for deprecated `try!()` macro

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
#[allow(unexpected_cfgs)]
fn cfg() {
    let _ = cfg!(pants);
    let _ = cfg!(pants,);
    let _ = cfg!(pants = "pants");
    let _ = cfg!(pants = "pants",);
    let _ = cfg!(all(pants));
    let _ = cfg!(all(pants),);
    let _ = cfg!(all(pants,));
    let _ = cfg!(all(pants,),);
}

#[test]
fn column() {
    let _ = column!();
}

// compile_error! is in a check-fail companion to this test

#[test]
fn concat() {
    let _ = concat!();
    let _ = concat!("hello");
    let _ = concat!("hello",);
    let _ = concat!("hello", " world");
    let _ = concat!("hello", " world",);
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
    let _ = env!("PATH");
    let _ = env!("PATH",);
    let _ = env!("PATH", "not found");
    let _ = env!("PATH", "not found",);
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
    let _ = file!();
}

#[cfg(std)]
#[test]
fn format() {
    let _ = format!("hello");
    let _ = format!("hello",);
    let _ = format!("hello {}", "world");
    let _ = format!("hello {}", "world",);
}

#[test]
fn format_args() {
    let _ = format_args!("hello");
    let _ = format_args!("hello",);
    let _ = format_args!("hello {}", "world");
    let _ = format_args!("hello {}", "world",);
}

#[test]
fn include() {
    include!("auxiliary/macro-comma-support.rs");
    include!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn include_bytes() {
    let _ = include_bytes!("auxiliary/macro-comma-support.rs");
    let _ = include_bytes!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn include_str() {
    let _ = include_str!("auxiliary/macro-comma-support.rs");
    let _ = include_str!("auxiliary/macro-comma-support.rs",);
}

#[test]
fn line() {
    let _ = line!();
}

#[test]
fn matches() {
    let _ = matches!(1, x if x > 0);
    let _ = matches!(1, x if x > 0,);
}

#[test]
fn module_path() {
    let _ = module_path!();
}

#[test]
fn option_env() {
    let _ = option_env!("PATH");
    let _ = option_env!("PATH",);
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
    let _ = vec![0];
    let _ = vec![0,];
    let _ = vec![0, 1];
    let _ = vec![0, 1,];
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
        let _ = write!(f, "hello");
        let _ = write!(f, "hello",);
        let _ = write!(f, "hello {}", "world");
        let _ = write!(f, "hello {}", "world",);
    }
}

test_with_formatter! {
    #[test]
    fn writeln(f: &mut fmt::Formatter) {
        let _ = writeln!(f);
        let _ = writeln!(f,);
        let _ = writeln!(f, "hello");
        let _ = writeln!(f, "hello",);
        let _ = writeln!(f, "hello {}", "world");
        let _ = writeln!(f, "hello {}", "world",);
    }
}
