//! Tests that compiler yields error E0191 when value with existing trait implementation
//! is cast as same `dyn` trait without specifying associated type at the cast.
//!
//! # Context
//! Original issue: https://github.com/rust-lang/rust/issues/21950

trait Add<Rhs=Self> {
    type Output;
}

impl Add for i32 {
    type Output = i32;
}

trait Meow {
    type Assoc;
}

struct Cat;

impl Meow for Cat {
    type Assoc = i32;
}

fn main() {
    let x = &10 as &dyn Add<i32, Output = i32>; //OK
    let x = &10 as &dyn Add;
    //~^ ERROR E0191

    // Regression test for https://github.com/rust-lang/rust/issues/155578.
    let cat = Cat;
    let _: &dyn Meow<> = &cat;
    //~^ ERROR E0191
}
