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
    //~^ non_canonical_partial_ord_impl
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

// lint, and give `_` a name

#[derive(Eq, PartialEq)]
struct C(u32);

impl Ord for C {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for C {
    //~^ non_canonical_partial_ord_impl
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        todo!();
    }
}

// do not lint derived

#[derive(Eq, Ord, PartialEq, PartialOrd)]
struct D(u32);

// do not lint if ord is not manually implemented

#[derive(Eq, PartialEq)]
struct E(u32);

impl PartialOrd for E {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!();
    }
}

// do not lint since ord has more restrictive bounds

#[derive(Eq, PartialEq)]
struct Uwu<A>(A);

impl<A: std::fmt::Debug + Ord + PartialOrd> Ord for Uwu<A> {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl<A: Ord + PartialOrd> PartialOrd for Uwu<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        todo!();
    }
}

// do not lint since `Rhs` is not `Self`

#[derive(Eq, PartialEq)]
struct F(u32);

impl Ord for F {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for F {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq<u32> for F {
    fn eq(&self, other: &u32) -> bool {
        todo!();
    }
}

impl PartialOrd<u32> for F {
    fn partial_cmp(&self, other: &u32) -> Option<Ordering> {
        todo!();
    }
}

// #11178, do not lint

#[derive(Eq, PartialEq)]
struct G(u32);

impl Ord for G {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for G {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Self::cmp(self, other))
    }
}

#[derive(Eq, PartialEq)]
struct H(u32);

impl Ord for H {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for H {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

// #12683, do not lint

#[derive(Eq, PartialEq)]
struct I(u32);

impl Ord for I {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for I {
    #[allow(clippy::needless_return)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

// #13640, do not lint

#[derive(Eq, PartialEq)]
struct J(u32);

impl Ord for J {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for J {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp(other).into()
    }
}

// #13640, check that a simple `.into()` does not obliterate the lint

#[derive(Eq, PartialEq)]
struct K(u32);

impl Ord for K {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for K {
    //~^ non_canonical_partial_ord_impl
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Ordering::Greater.into()
    }
}
