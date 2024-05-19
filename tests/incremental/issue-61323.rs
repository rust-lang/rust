//@ revisions: rpass cfail

enum A {
    //[cfail]~^ ERROR 3:1: 3:7: recursive types `A` and `C` have infinite size [E0072]
    B(C),
}

#[cfg(rpass)]
struct C(Box<A>);

#[cfg(cfail)]
struct C(A);

fn main() {}
