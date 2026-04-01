//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_local_definitions)]
// Various scenarios in which `pub` is required in blocks

struct S;

mod m {
    fn f() {
        impl crate::S {
            pub fn s(&self) {}
        }
    }
}

// Scenario 1

pub trait Tr {
    type A;
}
pub struct S1;

fn f() {
    pub struct Z;

    impl crate::Tr for crate::S1 {
        type A = Z; // Private-in-public error unless `struct Z` is pub
    }
}

// Scenario 2

trait Tr1 {
    type A;
    fn pull(&self) -> Self::A;
}
struct S2;

mod m1 {
    fn f() {
        pub struct Z {
            pub field: u8
        }

        impl crate::Tr1 for crate::S2 {
            type A = Z;
            fn pull(&self) -> Self::A { Z{field: 10} }
        }
    }
}

fn main() {
    S.s(); // Privacy error, unless `fn s` is pub
    let a = S2.pull().field; // Privacy error unless `field: u8` is pub
}
