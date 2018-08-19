// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test various cases where the defaults should lead to errors being
// reported.

#![allow(dead_code)]

trait SomeTrait {
    fn dummy(&self) { }
}

struct SomeStruct<'a> {
    r: Box<SomeTrait+'a>
}

fn load(ss: &mut SomeStruct) -> Box<SomeTrait> {
    // `Box<SomeTrait>` defaults to a `'static` bound, so this return
    // is illegal.

    ss.r //~ ERROR explicit lifetime required in the type of `ss` [E0621]
}

fn store(ss: &mut SomeStruct, b: Box<SomeTrait>) {
    // No error: b is bounded by 'static which outlives the
    // (anonymous) lifetime on the struct.

    ss.r = b;
}

fn store1<'b>(ss: &mut SomeStruct, b: Box<SomeTrait+'b>) {
    // Here we override the lifetimes explicitly, and so naturally we get an error.

    ss.r = b; //~ ERROR 41:12: 41:13: explicit lifetime required in the type of `ss` [E0621]
}

fn main() {
}
