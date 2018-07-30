// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(label_break_value)]

// These are forbidden occurences of label-break-value

fn labeled_unsafe() {
    unsafe 'b: {} //~ ERROR expected one of `extern`, `fn`, or `{`
}

fn labeled_if() {
    if true 'b: {} //~ ERROR expected `{`, found `'b`
}

fn labeled_else() {
    if true {} else 'b: {} //~ ERROR expected `{`, found `'b`
}

fn labeled_match() {
    match false 'b: {} //~ ERROR expected one of `.`, `?`, `{`, or an operator
}

pub fn main() {}
