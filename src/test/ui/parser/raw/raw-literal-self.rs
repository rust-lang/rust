// compile-flags: -Z parse-only

fn self_test(r#self: u32) {
    //~^ ERROR `r#self` is not currently supported.
}
