// https://github.com/rust-lang/rust/issues/54044
#![deny(unused_attributes)] //~ NOTE lint level is defined here

#[cold]
//~^ ERROR attribute cannot be used on
//~| WARN previously accepted
struct Foo;

fn main() {
    #[cold]
    //~^ ERROR attribute cannot be used on
    //~| WARN previously accepted
    5;
}
