//@ run-pass

#![feature(arbitrary_self_types, arbitrary_self_types_split_chains)]
#![allow(unused)]

use std::ops::{Deref, Receiver};

// Let us construct weird receiver chains.

//  ┌─────┐          ┌─────┐         ┌─────┐         ┌─────┐
//  │     │  Deref   │     │  Deref  │     │  Deref  │     │
//  │  A  ├──────────►  D  ├─────────►  C  ├─────────►  B  │
//  │     │          │     │         │     │         │     │
//  └──┬──┘          └──┬──┘         └─────┘         └──┬──┘
//     │                │                               │
//     │ Receiver       │ Receiver              Receiver│
//  ┌──▼──┐          ┌──▼──┐                         ┌──▼──┐
//  │     │          │     │                         │     │
//  │  B  │          │  A  │                         │  C  │
//  │     │          │     │                         │     │
//  └──┬──┘          └──┬──┘                         └─────┘
//     │ Receiver       │ Receiver
//  ┌──▼──┐          ┌──▼──┐
//  │     │          │     │
//  │  C  │          │  B  │
//  │     │          │     │
//  └─────┘          └──┬──┘
//                      │ Receiver
//                   ┌──▼──┐
//                   │     │
//                   │  C  │
//                   │     │
//                   └─────┘

struct A;
struct B;
struct C;
struct D;

impl Deref for A {
    type Target = D;
    fn deref(&self) -> &Self::Target {
        &D
    }
}

impl Deref for D {
    type Target = C;
    fn deref(&self) -> &Self::Target {
        &C
    }
}

impl Deref for C {
    type Target = B;
    fn deref(&self) -> &Self::Target {
        &B
    }
}

impl Receiver for A {
    type Target = B;
}

impl Receiver for B {
    type Target = C;
}

impl Receiver for D {
    type Target = A;
}

impl A {
    fn foo(self: &D) -> u8 {
        64
    }
    fn bar(self: &D) -> u8 {
        64
    }
}

impl B {
    fn foo(self: &D) -> u8 {
        88
    }
    fn bar(self: &B) -> u8 {
        88
    }
    fn boo(self: &B) -> u8 {
        88
    }
}

impl C {
    fn foo(self: &A) -> u8 {
        42
    }
    fn bar(self: &C) -> u8 {
        42
    }
    fn baz(self: &C) -> u8 {
        42
    }
    fn boo(self: &A, _: u8) -> u8 {
        88
    }
}

fn main() {
    let a = A;
    assert_eq!(a.foo(), 42); // This should be dispatched to C::foo
    assert_eq!(a.bar(), 64); // This should be dispatched to A::bar
    assert_eq!(a.baz(), 42); // This should be dispatched to C::baz
    assert_eq!((*a).boo(), 88); // This should be dispatched to B::boo
}
