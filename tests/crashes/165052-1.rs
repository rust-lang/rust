//@ known-bug: #165052

struct S<'a> {
    //~^ ERROR lifetime parameter `'a` is never used
    n: &'static u8, // changing 'static to 'a makes it compile
}
fn b(n: &u8) -> S {
    S { n }
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
