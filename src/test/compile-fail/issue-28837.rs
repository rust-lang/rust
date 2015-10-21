// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A;

fn main() {
    let a = A;

    a + a; //~ ERROR binary operation `+` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Add` might be missing for `A`

    a - a; //~ ERROR binary operation `-` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Sub` might be missing for `A`

    a * a; //~ ERROR binary operation `*` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Mul` might be missing for `A`

    a / a; //~ ERROR binary operation `/` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Div` might be missing for `A`

    a % a; //~ ERROR binary operation `%` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Rem` might be missing for `A`

    a & a; //~ ERROR binary operation `&` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::BitAnd` might be missing for `A`

    a | a; //~ ERROR binary operation `|` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::BitOr` might be missing for `A`

    a << a; //~ ERROR binary operation `<<` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Shl` might be missing for `A`

    a >> a; //~ ERROR binary operation `>>` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::ops::Shr` might be missing for `A`

    a == a; //~ ERROR binary operation `==` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialEq` might be missing for `A`

    a != a; //~ ERROR binary operation `!=` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialEq` might be missing for `A`

    a < a; //~ ERROR binary operation `<` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A`

    a <= a; //~ ERROR binary operation `<=` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A`

    a > a; //~ ERROR binary operation `>` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A`

    a >= a; //~ ERROR binary operation `>=` cannot be applied to type `A`
    //~^ NOTE an implementation of `std::cmp::PartialOrd` might be missing for `A`
}
