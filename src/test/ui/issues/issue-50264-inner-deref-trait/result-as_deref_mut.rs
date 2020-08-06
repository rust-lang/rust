fn main() {
    let _result = &mut Ok(42).as_deref_mut();
//~^ ERROR no method named `as_deref_mut` found
}
