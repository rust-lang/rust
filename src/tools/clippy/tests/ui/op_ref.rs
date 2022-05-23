#![allow(unused_variables, clippy::blacklisted_name)]
#![warn(clippy::op_ref)]
use std::collections::HashSet;
use std::ops::{BitAnd, Mul};

fn main() {
    let tracked_fds: HashSet<i32> = HashSet::new();
    let new_fds = HashSet::new();
    let unwanted = &tracked_fds - &new_fds;

    let foo = &5 - &6;

    let bar = String::new();
    let bar = "foo" == &bar;

    let a = "a".to_string();
    let b = "a";

    if b < &a {
        println!("OK");
    }

    struct X(i32);
    impl BitAnd for X {
        type Output = X;
        fn bitand(self, rhs: X) -> X {
            X(self.0 & rhs.0)
        }
    }
    impl<'a> BitAnd<&'a X> for X {
        type Output = X;
        fn bitand(self, rhs: &'a X) -> X {
            X(self.0 & rhs.0)
        }
    }
    let x = X(1);
    let y = X(2);
    let z = x & &y;

    #[derive(Copy, Clone)]
    struct Y(i32);
    impl BitAnd for Y {
        type Output = Y;
        fn bitand(self, rhs: Y) -> Y {
            Y(self.0 & rhs.0)
        }
    }
    impl<'a> BitAnd<&'a Y> for Y {
        type Output = Y;
        fn bitand(self, rhs: &'a Y) -> Y {
            Y(self.0 & rhs.0)
        }
    }
    let x = Y(1);
    let y = Y(2);
    let z = x & &y;
}

#[derive(Clone, Copy)]
struct A(i32);
#[derive(Clone, Copy)]
struct B(i32);

impl Mul<&A> for B {
    type Output = i32;
    fn mul(self, rhs: &A) -> Self::Output {
        self.0 * rhs.0
    }
}
impl Mul<A> for B {
    type Output = i32;
    fn mul(self, rhs: A) -> Self::Output {
        // Should not lint because removing the reference would lead to unconditional recursion
        self * &rhs
    }
}
impl Mul<&A> for A {
    type Output = i32;
    fn mul(self, rhs: &A) -> Self::Output {
        self.0 * rhs.0
    }
}
impl Mul<A> for A {
    type Output = i32;
    fn mul(self, rhs: A) -> Self::Output {
        let one = B(1);
        let two = 2;
        let three = 3;
        let _ = one * &self;
        let _ = two + &three;
        // Removing the reference would lead to unconditional recursion
        self * &rhs
    }
}
