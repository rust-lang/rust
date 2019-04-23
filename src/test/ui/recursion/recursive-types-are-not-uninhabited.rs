struct R<'a> {
    r: &'a R<'a>,
}

fn foo(res: Result<u32, &R>) -> u32 {
    let Ok(x) = res;
    //~^ ERROR refutable pattern
    x
    //~^ WARN use of possibly uninitialized variable: `x`
    //~| WARN this error has been downgraded to a warning for backwards compatibility
    //~| WARN this represents potential undefined behavior in your code and this warning will
}

fn main() {
    foo(Ok(23));
}
