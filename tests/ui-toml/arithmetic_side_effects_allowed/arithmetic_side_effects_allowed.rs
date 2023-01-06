#![warn(clippy::arithmetic_side_effects)]

use core::ops::{Add, Neg};

macro_rules! create {
    ($name:ident) => {
        #[allow(clippy::arithmetic_side_effects)]
        #[derive(Clone, Copy)]
        struct $name;

        impl Add<$name> for $name {
            type Output = $name;
            fn add(self, other: $name) -> Self::Output {
                todo!()
            }
        }

        impl Add<i32> for $name {
            type Output = $name;
            fn add(self, other: i32) -> Self::Output {
                todo!()
            }
        }

        impl Add<$name> for i32 {
            type Output = $name;
            fn add(self, other: $name) -> Self::Output {
                todo!()
            }
        }

        impl Add<i64> for $name {
            type Output = $name;
            fn add(self, other: i64) -> Self::Output {
                todo!()
            }
        }

        impl Add<$name> for i64 {
            type Output = $name;
            fn add(self, other: $name) -> Self::Output {
                todo!()
            }
        }

        impl Neg for $name {
            type Output = $name;
            fn neg(self) -> Self::Output {
                todo!()
            }
        }
    };
}

create!(Foo);
create!(Bar);
create!(Baz);
create!(OutOfNames);

fn lhs_and_rhs_are_equal() {
    // is explicitly on the list
    let _ = OutOfNames + OutOfNames;
    // is explicitly on the list
    let _ = Foo + Foo;
    // is implicitly on the list
    let _ = Bar + Bar;
    // not on the list
    let _ = Baz + Baz;
}

fn lhs_is_different() {
    // is explicitly on the list
    let _ = 1i32 + OutOfNames;
    // is explicitly on the list
    let _ = 1i32 + Foo;
    // is implicitly on the list
    let _ = 1i32 + Bar;
    // not on the list
    let _ = 1i32 + Baz;

    // not on the list
    let _ = 1i64 + Foo;
    // is implicitly on the list
    let _ = 1i64 + Bar;
    // not on the list
    let _ = 1i64 + Baz;
}

fn rhs_is_different() {
    // is explicitly on the list
    let _ = OutOfNames + 1i32;
    // is explicitly on the list
    let _ = Foo + 1i32;
    // is implicitly on the list
    let _ = Bar + 1i32;
    // not on the list
    let _ = Baz + 1i32;

    // not on the list
    let _ = Foo + 1i64;
    // is implicitly on the list
    let _ = Bar + 1i64;
    // not on the list
    let _ = Baz + 1i64;
}

fn unary() {
    // is explicitly on the list
    let _ = -OutOfNames;
    // is explicitly on the list
    let _ = -Foo;
    // not on the list
    let _ = -Bar;
    // not on the list
    let _ = -Baz;
}

fn main() {}
