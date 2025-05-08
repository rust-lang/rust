// If it is impossible to find query arguments just from the hash
// compiler should treat the node as red

// In this test prior to fixing compiler was having problems figuring out
// drop impl for T inside of m

//@ revisions:cfail1 cfail2
//@ compile-flags: --crate-type=lib
//@ build-pass

#![allow(dead_code)]

pub trait P {
    type A;
}

struct S;

impl P for S {
    type A = C;
}

struct T<D: P>(D::A, Z<D>);

struct Z<D: P>(D::A, String);

impl<D: P> T<D> {
    pub fn i() -> Self {
        loop {}
    }
}

enum C {
    #[cfg(cfail1)]
    Up(()),
    #[cfg(cfail2)]
    Lorry(()),
}

pub fn m() {
    T::<S>::i();
}
