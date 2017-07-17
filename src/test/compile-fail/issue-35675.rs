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
    Orange(i64),
}

fn should_return_fruit() -> Apple {
    //~^ ERROR cannot find type `Apple` in this scope
    //~| NOTE not found in this scope
    //~| HELP you can try using the variant's enum
    Apple(5)
    //~^ ERROR cannot find function `Apple` in this scope
    //~| NOTE not found in this scope
}

fn should_return_fruit_too() -> Fruit::Apple {
    //~^ ERROR expected type, found variant `Fruit::Apple`
    //~| HELP you can try using the variant's enum
    //~| NOTE not a type
    Apple(5)
    //~^ ERROR cannot find function `Apple` in this scope
    //~| NOTE not found in this scope
}

fn foo() -> Ok {
    //~^ ERROR expected type, found variant `Ok`
    //~| NOTE not a type
    //~| HELP there is an enum variant
    //~| HELP there is an enum variant
    Ok(())
}

fn bar() -> Variant3 {
    //~^ ERROR cannot find type `Variant3` in this scope
    //~| HELP you can try using the variant's enum
    //~| NOTE not found in this scope
}

fn qux() -> Some {
    //~^ ERROR expected type, found variant `Some`
    //~| NOTE not a type
    //~| HELP there is an enum variant
    //~| HELP there is an enum variant
    Some(1)
}

fn main() {}

mod x {
    enum Enum {
        Variant1,
        Variant2(),
        Variant3(usize),
        Variant4 {},
    }
}
