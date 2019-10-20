fn main() {
    let _result = &mut Some(42).as_deref_mut();
//~^ ERROR no method named `as_deref_mut` found for type `std::option::Option<{integer}>`
}
