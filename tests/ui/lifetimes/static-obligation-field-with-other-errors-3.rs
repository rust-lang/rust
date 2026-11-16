// Regression test for #165052
// This used to ICE
//@ check-fail

struct S<'a> {
    m: &'a dyn std::fmt::Debug,
    n: &'static u8,
}
fn b<'a>(
    m: &'a dyn std::fmt::Debug,
    n: &'a u8,
) -> S<'a> {
    S {
        m,
        n,
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {}
