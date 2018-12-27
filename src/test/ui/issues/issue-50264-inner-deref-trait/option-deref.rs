#![feature(inner_deref)]

fn main() {
    let _result = &Some(42).deref();
//~^ ERROR no method named `deref` found for type `std::option::Option<{integer}>`
}
