#![allow(unused)]
#![warn(clippy::missing_asserts_for_indexing)]

fn sum(v: &[u8]) -> u8 {
    v[0] + v[1] + v[2] + v[3] + v[4]
    //~^ ERROR: indexing into a slice multiple times without an `assert`
}

fn subslice(v: &[u8]) {
    let _ = v[0];
    //~^ ERROR: indexing into a slice multiple times without an `assert`
    let _ = v[1..4];
}

fn variables(v: &[u8]) -> u8 {
    let a = v[0];
    //~^ ERROR: indexing into a slice multiple times without an `assert`
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
    v2: &'a [u8],
}

fn index_struct_field(f: &Foo<'_>) {
    let _ = f.v[0] + f.v[1];
    //~^ ERROR: indexing into a slice multiple times without an `assert`
}

fn index_struct_different_fields(f: &Foo<'_>) {
    // ok, different fields
    let _ = f.v[0] + f.v2[1];
}

fn shadowing() {
    let x: &[i32] = &[1];
    assert!(x.len() > 1);

    let x: &[i32] = &[1];
    let _ = x[0] + x[1];
    //~^ ERROR: indexing into a slice multiple times without an `assert`
}

pub fn issue11856(values: &[i32]) -> usize {
    let mut ascending = Vec::new();
    for w in values.windows(2) {
        assert!(w.len() > 1);
        if w[0] < w[1] {
            ascending.push((w[0], w[1]));
        } else {
            ascending.push((w[1], w[0]));
        }
    }
    ascending.len()
}

fn main() {}
