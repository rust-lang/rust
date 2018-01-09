// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]
#![allow(dead_code)]
#![allow(path_statements)]
#![allow(unreachable_patterns)]

fn never_direct(x: !) {
    x;
}

fn never_ref_pat(ref x: !) {
    *x;
}

fn never_ref(x: &!) {
    let &y = x;
    y;
}

fn never_slice(x: &[!]) {
    x[0];
}

fn never_match(x: Result<(), !>) {
    match x {
        Ok(_) => {},
        Err(_) => {},
    }
}

pub fn main() { }

// END RUST SOURCE

// START rustc.never_direct.SimplifyCfg-initial.after.mir
// bb0: {
//     unreachable;
// }
// END rustc.never_direct.SimplifyCfg-initial.after.mir

// START rustc.never_ref_pat.SimplifyCfg-initial.after.mir
// bb0: {
//     unreachable;
// }
// END rustc.never_ref_pat.SimplifyCfg-initial.after.mir

// START rustc.never_ref.SimplifyCfg-initial.after.mir
// bb0: {
//     unreachable;
// }
// END rustc.never_ref.SimplifyCfg-initial.after.mir

// START rustc.never_slice.SimplifyCfg-initial.after.mir
// bb1: {
//     ...
//     unreachable;
// }
// END rustc.never_slice.SimplifyCfg-initial.after.mir

// START rustc.never_match.SimplifyCfg-initial.after.mir
// fn never_match(_1: std::result::Result<(), !>) -> () {
//     ...
//     bb0: {
//         ...
//     }
//     bb1: {
//         unreachable;
//     }
//     bb2: {
//         _0 = ();
//         return;
//     }
// }
// END rustc.never_match.SimplifyCfg-initial.after.mir
