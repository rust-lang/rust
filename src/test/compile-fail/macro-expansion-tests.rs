// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod macros_cant_escape_fns {
    fn f() {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() } //~ ERROR macro undefined
}

mod macros_cant_escape_mods {
    mod f {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() } //~ ERROR macro undefined
}

mod macros_can_escape_flattened_mods_test {
    #[macro_use]
    mod f {
        macro_rules! m { () => { 3 + 4 } }
    }
    fn g() -> i32 { m!() }
}

fn macro_tokens_should_match() {
    macro_rules! m { (a) => { 13 } }
    m!(a);
}

// should be able to use a bound identifier as a literal in a macro definition:
fn self_macro_parsing() {
    macro_rules! foo { (zz) => { 287; } }
    fn f(zz: i32) {
        foo!(zz);
    }
}

fn main() {}
