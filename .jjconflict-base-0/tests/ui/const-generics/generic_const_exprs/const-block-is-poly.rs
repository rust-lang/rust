#![feature(generic_const_exprs)]

fn foo<T>() {
    let _ = [0u8; { const { std::mem::size_of::<T>() } }];
    //~^ ERROR: overly complex generic constant
}

fn main() {
    foo::<i32>();
}
