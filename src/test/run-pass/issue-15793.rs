// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum NestedEnum {
    First,
    Second,
    Third
}
enum Enum {
    Variant1(bool),
    Variant2(NestedEnum)
}

#[inline(never)]
fn foo(x: Enum) -> int {
    match x {
        Enum::Variant1(true) => 1,
        Enum::Variant1(false) => 2,
        Enum::Variant2(NestedEnum::Second) => 3,
        Enum::Variant2(NestedEnum::Third) => 4,
        Enum::Variant2(NestedEnum::First) => 5
    }
}

fn main() {
    assert_eq!(foo(Enum::Variant2(NestedEnum::Third)), 4);
}
