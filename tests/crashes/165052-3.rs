//@ known-bug: #165052

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
