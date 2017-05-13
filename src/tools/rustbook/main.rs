// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

#![feature(rustc_private)]
#![feature(rustdoc)]

extern crate rustdoc;
extern crate rustc_back;

use std::env;
use std::process;
use std::sync::atomic::{AtomicIsize, ATOMIC_ISIZE_INIT, Ordering};
use term::Term;

mod term;
mod error;
mod book;

mod subcommand;
mod help;
mod build;
mod serve;
mod test;

static EXIT_STATUS: AtomicIsize = ATOMIC_ISIZE_INIT;

pub fn main() {
    let mut term = Term::new();
    let cmd: Vec<_> = env::args().collect();

    if cmd.len() <= 1 {
        help::usage()
    } else {
        match subcommand::parse_name(&cmd[1][..]) {
            Some(mut subcmd) => {
                match subcmd.parse_args(&cmd[..cmd.len()-1]) {
                    Ok(_) => {
                        match subcmd.execute(&mut term) {
                            Ok(_) => (),
                            Err(err) => {
                                term.err(&format!("error: {}", err));
                            }
                        }
                    }
                    Err(err) => {
                        println!("{}", err.description());
                        println!("");
                        subcmd.usage();
                    }
                }
            }
            None => {
                println!("Unrecognized command '{}'.", cmd[1]);
                println!("");
                help::usage();
            }
        }
    }
    process::exit(EXIT_STATUS.load(Ordering::SeqCst) as i32);
}
