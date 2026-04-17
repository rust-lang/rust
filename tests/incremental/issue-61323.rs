//@ revisions: rpass bfail

enum A {
    //[bfail]~^ ERROR recursive types `A` and `C` have infinite size [E0072]
    B(C),
}

#[cfg(rpass)]
struct C(Box<A>);

#[cfg(bfail)]
struct C(A);

fn main() {}
