fn main() {
    match Some("hi".to_string()) {
        ref op_string_ref @ Some(s) => {},
        //~^ ERROR pattern bindings are not allowed after an `@` [E0303]
        //~| ERROR E0009
        None => {},
    }
}
