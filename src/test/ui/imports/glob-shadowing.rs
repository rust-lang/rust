// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(decl_macro)]

mod m {
    pub macro env($e: expr) { $e }
    pub macro fenv() { 0 }
}

mod glob_in_normal_module {
    use m::*;
    fn check() {
        let x = env!("PATH"); //~ ERROR `env` is ambiguous
    }
}

mod glob_in_block_module {
    fn block() {
        use m::*;
        fn check() {
            let x = env!("PATH"); //~ ERROR `env` is ambiguous
        }
    }
}

mod glob_shadows_item {
    pub macro fenv($e: expr) { $e }
    fn block() {
        use m::*;
        fn check() {
            let x = fenv!(); //~ ERROR `fenv` is ambiguous
        }
    }
}

fn main() {}
