#![allow(unused)]
#![warn(clippy::manual_partial_ord_impl)]
#![no_main]

use std::cmp::Ordering;

// lint

#[derive(Eq, PartialEq)]
struct A(u32);

impl Ord for A {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for A {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!();
    }
}

// do not lint

#[derive(Eq, PartialEq)]
struct B(u32);

impl Ord for B {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for B {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// lint, but we cannot give a suggestion since &Self is not named

#[derive(Eq, PartialEq)]
struct C(u32);

impl Ord for C {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for C {
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        todo!();
    }
}

