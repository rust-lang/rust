// compile-flags: -Z parse-only

fn underscore_test(r#_: u32) {
    //~^ ERROR `r#_` is not currently supported.
}
