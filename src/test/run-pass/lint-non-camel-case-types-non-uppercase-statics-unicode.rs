// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast feature doesn't work.

#[forbid(non_camel_case_types)];
#[forbid(non_uppercase_statics)];
#[feature(non_ascii_idents)];

// Some scripts (e.g. hiragana) don't have a concept of
// upper/lowercase

struct ヒ;

static ラ: uint = 0;

pub fn main() {}
