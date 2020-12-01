fn main() {
    let _result = &Some(42).as_deref();
//~^ ERROR no method named `as_deref` found for enum `Option<{integer}>`
}
