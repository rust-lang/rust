#![allow(unused)]
#![warn(clippy::missing_asserts_for_indexing)]

fn sum(v: &[u8]) -> u8 {
    //~^ ERROR missing assertions on `v.len()`
    v[0] + v[1] + v[2] + v[3] + v[4]
}

fn subslice(v: &[u8]) {
    //~^ ERROR missing assertion on `v.len()`
    let _ = v[0];
    let _ = v[1..4];
}

fn variables(v: &[u8]) -> u8 {
    //~^ ERROR missing assertions on `v.len()`
    let a = v[0];
    let b = v[1];
    let c = v[2];
    a + b + c
}

fn index_different_slices(v1: &[u8], v2: &[u8]) {
    let _ = v1[0] + v1[12];
    let _ = v2[5] + v2[15];
}

fn index_different_slices2(v1: &[u8], v2: &[u8]) {
    assert!(v1.len() > 12);
    let _ = v1[0] + v1[12];
    let _ = v2[5] + v2[15];
}

struct Foo<'a> {
    v: &'a [u8],
}

fn index_struct_field(f: &Foo<'_>) {
    //~^ ERROR missing assertion on `f.v.len()`
    let _ = f.v[0] + f.v[1];
}

fn main() {}
