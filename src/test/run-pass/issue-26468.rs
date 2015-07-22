// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

enum FooMode {
    Check = 0x1001,
}

enum BarMode {
    Check = 0x2001,
}

enum Mode {
    Foo(FooMode),
    Bar(BarMode),
}

#[inline(never)]
fn broken(mode: &Mode) -> u32 {
    for _ in 0..1 {
        if let Mode::Foo(FooMode::Check) = *mode { return 17 }
        if let Mode::Bar(BarMode::Check) = *mode { return 19 }
    }
    return 42;
}

fn main() {
    let mode = Mode::Bar(BarMode::Check);
    assert_eq!(broken(&mode), 19);
}
