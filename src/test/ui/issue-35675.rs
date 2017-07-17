// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// these two HELPs are actually in a new line between this line and the `enum Fruit` line
enum Fruit { //~ HELP possible candidate is found in another module, you can import it into scope
    //~^ HELP possible candidate is found in another module, you can import it into scope
    Apple(i64),
    //~^ HELP there is an enum variant `Fruit::Apple`, did you mean to use `Fruit`?
    //~| HELP there is an enum variant `Fruit::Apple`, did you mean to use `Fruit`?
    Orange(i64),
}

fn should_return_fruit() -> Apple {
    //~^ ERROR cannot find type `Apple` in this scope
    //~| NOTE not found in this scope
    Apple(5)
    //~^ ERROR cannot find function `Apple` in this scope
    //~| NOTE not found in this scope
}

fn should_return_fruit_too() -> Fruit::Apple {
    //~^ ERROR expected type, found variant `Fruit::Apple`
    //~| NOTE not a type
    Apple(5)
    //~^ ERROR cannot find function `Apple` in this scope
    //~| NOTE not found in this scope
}

fn bar() -> Variant3 {
    //~^ ERROR cannot find type `Variant3` in this scope
    //~| NOTE not found in this scope
}

fn main() {}

mod x {
    enum Enum {
        Variant1,
        Variant2(),
        Variant3(usize),
        //~^ HELP there is an enum variant `x::Enum::Variant3`, did you mean to use `x::Enum`?
        Variant4 {},
    }
}
