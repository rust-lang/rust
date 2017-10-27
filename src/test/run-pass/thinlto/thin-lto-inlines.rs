// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z thinlto -C codegen-units=8 -O
// min-llvm-version 4.0
// ignore-emscripten can't inspect instructions on emscripten

// We want to assert here that ThinLTO will inline across codegen units. There's
// not really a great way to do that in general so we sort of hack around it by
// praying two functions go into separate codegen units and then assuming that
// if inlining *doesn't* happen the first byte of the functions will differ.

pub fn foo() -> u32 {
    bar::bar()
}

mod bar {
    pub fn bar() -> u32 {
        3
    }
}

fn main() {
    println!("{} {}", foo(), bar::bar());

    unsafe {
        let foo = foo as usize as *const u8;
        let bar = bar::bar as usize as *const u8;

        assert_eq!(*foo, *bar);
    }
}
