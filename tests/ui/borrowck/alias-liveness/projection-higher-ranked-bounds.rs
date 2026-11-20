//@ check-pass
//@ known-bug: unknown
//@ edition: 2024

// We need to treat args as live if an impl *could* supply the bounds that allow
// them to end up in the hidden type.

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

// `for<'a, 'b, 'c> Tr<'a, 'b, 'c>` could mean that we have an impl where
// `'c: 'a + 'b`, so we should conservatively treat `'c` as live
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
