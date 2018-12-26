// compile-flags: -Z parse-only

impl S {
    fn f(*, a: u8) -> u8 {}
    //~^ ERROR expected argument name, found `*`
}
