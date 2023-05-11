struct R<'a> {
    r: &'a R<'a>,
}

fn foo(res: Result<u32, &R>) -> u32 {
    let Ok(x) = res;
    //~^ ERROR refutable pattern
    x
}

fn main() {
    foo(Ok(23));
}
