//@ check-pass
//@ edition: 2024

// This is another test for precision of lifetimes held within an associated
// type. Here, because we are returning an associated type directly, we can
// enumerate the bounds to discard unrelated lifetimes.

#![forbid(unsafe_code)]

use std::fmt::Debug;

trait Tr<'a, 'b, 'c> {
    type A: Debug + 'a + 'b;
    fn mk(c: &'c u8) -> Self::A;
}

fn run<'a, 'b, 'c, T: Tr<'a, 'b, 'c>>(c: &'c u8) -> T::A {
    T::mk(c)
}

fn show<T: Debug>(t: T, r: &Vec<u8>) {
    println!("escaped = {t:?}, reused = {r:?}");
}

fn test<T: for<'a, 'b, 'c> Tr<'a, 'b, 'c>>() {
    let escaped;
    {
        let v: Vec<u8> = vec![42];
        escaped = run::<'_, '_, '_, T>(&v[0]);
    }
    let reused: Vec<u8> = vec![7];
    show(escaped, &reused);
}

fn main() {}
