#![feature(min_const_generics)]

use std::mem::size_of;

fn test<const N: usize>() {}

fn ok<const M: usize>() -> [u8; M] {
    [0; { M }]
}

struct Break0<const N: usize>([u8; { N + 1 }]);
//~^ ERROR generic parameters may not be used in const operations

struct Break1<const N: usize>([u8; { { N } }]);
//~^ ERROR generic parameters may not be used in const operations

fn break2<const N: usize>() {
    let _: [u8; N + 1];
    //~^ ERROR generic parameters may not be used in const operations
}

fn break3<const N: usize>() {
    let _ = [0; N + 1];
    //~^ ERROR generic parameters may not be used in const operations
}

struct BreakTy0<T>(T, [u8; { size_of::<*mut T>() }]);
//~^ ERROR generic parameters may not be used in const operations

struct BreakTy1<T>(T, [u8; { { size_of::<*mut T>() } }]);
//~^ ERROR generic parameters may not be used in const operations

fn break_ty2<T>() {
    let _: [u8; size_of::<*mut T>() + 1];
    //~^ ERROR generic parameters may not be used in const operations
}

fn break_ty3<T>() {
    let _ = [0; size_of::<*mut T>() + 1];
    //~^ WARN cannot use constants which depend on generic parameters in types
    //~| WARN this was previously accepted by the compiler but is being phased out
}


trait Foo {
    const ASSOC: usize;
}

fn main() {}
