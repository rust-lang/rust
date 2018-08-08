// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Companion test to the similarly-named file in run-pass.

// compile-flags: -C debug_assertions=yes
// revisions: std core

#![cfg_attr(core, no_std)]

#[cfg(std)] use std::fmt;
#[cfg(core)] use core::fmt;

// (see documentation of the similarly-named test in run-pass)
fn to_format_or_not_to_format() {
    let falsum = || false;

    // assert!(true, "{}",); // see run-pass

    assert_eq!(1, 1, "{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments
    assert_ne!(1, 2, "{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments

    // debug_assert!(true, "{}",); // see run-pass

    debug_assert_eq!(1, 1, "{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments
    debug_assert_ne!(1, 2, "{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments

    #[cfg(std)] {
        eprint!("{}",);
        //[std]~^ ERROR no arguments
    }

    #[cfg(std)] {
        // FIXME: compile-fail says "expected error not found" even though
        //        rustc does emit an error
        // eprintln!("{}",);
        // <DISABLED> [std]~^ ERROR no arguments
    }

    #[cfg(std)] {
        format!("{}",);
        //[std]~^ ERROR no arguments
    }

    format_args!("{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments

    // if falsum() { panic!("{}",); } // see run-pass

    #[cfg(std)] {
        print!("{}",);
        //[std]~^ ERROR no arguments
    }

    #[cfg(std)] {
        // FIXME: compile-fail says "expected error not found" even though
        //        rustc does emit an error
        // println!("{}",);
        // <DISABLED> [std]~^ ERROR no arguments
    }

    unimplemented!("{}",);
    //[core]~^ ERROR no arguments
    //[std]~^^ ERROR no arguments

    // if falsum() { unreachable!("{}",); } // see run-pass

    struct S;
    impl fmt::Display for S {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}",)?;
            //[core]~^ ERROR no arguments
            //[std]~^^ ERROR no arguments

            // FIXME: compile-fail says "expected error not found" even though
            //        rustc does emit an error
            // writeln!(f, "{}",)?;
            // <DISABLED> [core]~^ ERROR no arguments
            // <DISABLED> [std]~^^ ERROR no arguments
            Ok(())
        }
    }
}

fn main() {}
