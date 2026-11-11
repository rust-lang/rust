// Regression test for #165052
// This used to ICE
//@ check-fail
struct S<'a> {
    //~^ ERROR lifetime parameter `'a` is never used
    n: &'static u8, // changing 'static to 'a makes it compile
}
fn b(n: &u8) -> S {
    S { n }
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
