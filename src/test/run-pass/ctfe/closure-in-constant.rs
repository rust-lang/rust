// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo ([u8; (|x: u8| { }, 9).1]);

// FIXME(#50689) We'd like the below to also work,
// but due to how DefCollector handles discr_expr differently it doesn't right now.
/*
enum Functions {
    Square = (|x:i32| { }, 42).1,
}
*/

fn main() {}
