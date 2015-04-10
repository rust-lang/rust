// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z force-dropflag-checks=off

// Quick-and-dirty test to ensure -Z force-dropflag-checks=off works as
// expected. Note that the inlined drop-flag is slated for removal
// (RFC 320); when that happens, the -Z flag and this test should
// simply be removed.
//
// See also drop-flag-sanity-check.rs.

#![feature(old_io)]

use std::env;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "test" {
        return test();
    }

    let mut p = Command::new(&args[0]).arg("test").spawn().unwrap();
    // Invocatinn should succeed as drop-flag sanity check is skipped.
    assert!(p.wait().unwrap().success());
}

#[derive(Debug)]
struct Corrupted {
    x: u8
}

impl Drop for Corrupted {
    fn drop(&mut self) { println!("dropping"); }
}

fn test() {
    {
        let mut c1 = Corrupted { x: 1 };
        let mut c2 = Corrupted { x: 2 };
        unsafe {
            let p1 = &mut c1 as *mut Corrupted as *mut u8;
            let p2 = &mut c2 as *mut Corrupted as *mut u8;
            for i in 0..std::mem::size_of::<Corrupted>() {
                // corrupt everything, *including the drop flag.
                //
                // (We corrupt via two different means to safeguard
                // against the hypothetical assignment of the
                // dtor_needed/dtor_done values to v and v+k.  that
                // happen to match with one of the corruption values
                // below.)
                *p1.offset(i as isize) += 2;
                *p2.offset(i as isize) += 3;
            }
        }
        // Here, at the end of the scope of `c1` and `c2`, the
        // drop-glue should detect the corruption of (at least one of)
        // the drop-flags.
    }
    println!("We should never get here.");
}
