//! ICE: https://github.com/rust-lang/rust/issues/114121
//! This test checks that MIR validation never constrains
//! new hidden types that *differ* from the actual hidden types.
//! This test used to ICE because oli-obk assumed mir validation
//! was only ever run after opaque types were revealed in MIR.

//@ compile-flags: -Zvalidate-mir
//@ check-pass

fn main() {
    let _ = Some(()).into_iter().flat_map(|_| Some(()).into_iter().flat_map(func));
}

fn func(_: ()) -> impl Iterator<Item = ()> {
    Some(()).into_iter().flat_map(|_| vec![])
}
