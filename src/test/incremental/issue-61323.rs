// revisions: rpass cfail

enum A {
    //[cfail]~^ ERROR 3:1: 3:7: recursive type `A` has infinite size [E0072]
    B(C),
}

#[cfg(rpass)]
struct C(Box<A>);

#[cfg(cfail)]
struct C(A);
//[cfail]~^ ERROR 12:1: 12:13: recursive type `C` has infinite size [E0072]

fn main() {}
