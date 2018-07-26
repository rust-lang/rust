// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a test of the borrowck migrate mode. It leverages #27282, a
// bug that is fixed by NLL: this code is (unsoundly) accepted by
// AST-borrowck, but is correctly rejected by the NLL borrowck.
//
// Therefore, for backwards-compatiblity, under borrowck=migrate the
// NLL checks will be emitted as *warnings*.

// NLL mode makes this compile-fail; we cannot currently encode a
// test that is run-pass or compile-fail based on compare-mode. So
// just ignore it instead:

// ignore-compare-mode-nll

// revisions: zflag edition
//[zflag]compile-flags: -Z borrowck=migrate
//[edition]compile-flags: --edition 2018
//[zflag] run-pass
//[edition] run-pass

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if {
                (|| { let bar = foo; bar.take() })();
                false
            } => {},
        Some(ref _s) => println!("Note this arm is bogus; the `Some` became `None` in the guard."),
        _ => println!("Here is some supposedly unreachable code."),
    }
}
