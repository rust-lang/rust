#![allow(unused)]
#![warn(clippy::missing_asserts_for_indexing)]

// ok
fn sum_with_assert(v: &[u8]) -> u8 {
    assert!(v.len() > 4);
    v[0] + v[1] + v[2] + v[3] + v[4]
}

// ok
fn sum_with_assert_other_way(v: &[u8]) -> u8 {
    assert!(5 <= v.len());
    v[0] + v[1] + v[2] + v[3] + v[4]
}

// ok
fn sum_with_assert_ge(v: &[u8]) -> u8 {
    assert!(v.len() >= 5);
    v[0] + v[1] + v[2] + v[3] + v[4]
}

// ok
fn sum_with_assert_ge_other_way(v: &[u8]) -> u8 {
    assert!(4 < v.len());
    v[0] + v[1] + v[2] + v[3] + v[4]
}

fn sum_with_assert_lt(v: &[u8]) -> u8 {
    assert!(v.len() < 5);
    v[0] + v[1] + v[2] + v[3] + v[4]
    //~^ missing_asserts_for_indexing
}

fn sum_with_assert_le(v: &[u8]) -> u8 {
    assert!(v.len() <= 5);
    v[0] + v[1] + v[2] + v[3] + v[4]
    //~^ missing_asserts_for_indexing
}

fn sum_with_incorrect_assert_len(v: &[u8]) -> u8 {
    assert!(v.len() > 3);
    v[0] + v[1] + v[2] + v[3] + v[4]
    //~^ missing_asserts_for_indexing
}

fn sum_with_incorrect_assert_len2(v: &[u8]) -> u8 {
    assert!(v.len() >= 4);
    v[0] + v[1] + v[2] + v[3] + v[4]
    //~^ missing_asserts_for_indexing
}

// ok, don't lint for single array access
fn single_access(v: &[u8]) -> u8 {
    v[0]
}

// ok
fn subslice_ok(v: &[u8]) {
    assert!(v.len() > 3);
    let _ = v[0];
    let _ = v[1..4];
}

fn subslice_bad(v: &[u8]) {
    assert!(v.len() >= 3);
    let _ = v[0];
    //~^ missing_asserts_for_indexing

    let _ = v[1..4];
}

// ok
fn subslice_inclusive_ok(v: &[u8]) {
    assert!(v.len() > 4);
    let _ = v[0];
    let _ = v[1..=4];
}

fn subslice_inclusive_bad(v: &[u8]) {
    assert!(v.len() >= 4);
    let _ = v[0];
    //~^ missing_asserts_for_indexing

    let _ = v[1..=4];
}

fn index_different_slices_ok(v1: &[u8], v2: &[u8]) {
    assert!(v1.len() > 12);
    assert!(v2.len() > 15);
    let _ = v1[0] + v1[12];
    let _ = v2[5] + v2[15];
}

fn index_different_slices_wrong_len(v1: &[u8], v2: &[u8]) {
    assert!(v1.len() >= 12);
    assert!(v2.len() >= 15);
    let _ = v1[0] + v1[12];
    //~^ missing_asserts_for_indexing

    let _ = v2[5] + v2[15];
    //~^ missing_asserts_for_indexing
}
fn index_different_slices_one_wrong_len(v1: &[u8], v2: &[u8]) {
    assert!(v1.len() >= 12);
    assert!(v2.len() > 15);
    let _ = v1[0] + v1[12];
    //~^ missing_asserts_for_indexing

    let _ = v2[5] + v2[15];
}

fn side_effect() -> &'static [u8] {
    &[]
}

fn index_side_effect_expr() {
    let _ = side_effect()[0] + side_effect()[1];
}

// ok, single access for different slices
fn index_different_slice_in_same_expr(v1: &[u8], v2: &[u8]) {
    let _ = v1[0] + v2[1];
}

fn issue11835(v1: &[u8], v2: &[u8], v3: &[u8], v4: &[u8]) {
    assert!(v1.len() == 2);
    assert!(v2.len() == 4);
    assert!(2 == v3.len());
    assert!(4 == v4.len());

    let _ = v1[0] + v1[1] + v1[2];
    //~^ missing_asserts_for_indexing

    let _ = v2[0] + v2[1] + v2[2];

    let _ = v3[0] + v3[1] + v3[2];
    //~^ missing_asserts_for_indexing

    let _ = v4[0] + v4[1] + v4[2];
}

// ok
fn same_index_multiple_times(v1: &[u8]) {
    let _ = v1[0] + v1[0];
}

// ok
fn highest_index_first(v1: &[u8]) {
    let _ = v1[2] + v1[1] + v1[0];
}

fn issue14255(v1: &[u8], v2: &[u8], v3: &[u8], v4: &[u8]) {
    assert_eq!(v1.len(), 2);
    assert_eq!(v2.len(), 4);
    assert_eq!(2, v3.len());
    assert_eq!(4, v4.len());

    let _ = v1[0] + v1[1] + v1[2];
    //~^ missing_asserts_for_indexing

    let _ = v2[0] + v2[1] + v2[2];

    let _ = v3[0] + v3[1] + v3[2];
    //~^ missing_asserts_for_indexing

    let _ = v4[0] + v4[1] + v4[2];
}

fn main() {}
