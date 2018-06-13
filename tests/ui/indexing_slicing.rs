#![feature(plugin)]
#![warn(indexing_slicing)]
#![warn(out_of_bounds_indexing)]
#![allow(no_effect, unnecessary_operation)]

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    let index_from: usize = 2;
    let index_to: usize = 3;
    x[index];
    &x[index_from..index_to];
    &x[index_from..][..index_to];
    &x[index..];
    &x[..index];
    x[0];
    x[3];
    x[4];
    x[1 << 3];
    &x[1..5];
    &x[1..][..5];
    &x[0..3];
    &x[0..][..3];
    &x[0..].get(..3); // Ok
    &x[0..=4];
    &x[..=4];
    &x[..];
    &x[1..];
    &x[4..];
    &x[5..];
    &x[..4];
    &x[..5];

    let y = &x;
    y[0];
    &y[1..2];
    &y[..];
    &y[0..=4];
    &y[..=4];

    let empty: [i8; 0] = [];
    empty[0];
    &empty[1..5];
    &empty[0..=4];
    &empty[..=4];
    &empty[..];
    &empty[0..];
    &empty[0..0];
    &empty[0..=0];
    &empty[..=0];
    &empty[..0];
    &empty[1..];
    &empty[..4];

    let v = vec![0; 5];
    v[0];
    v[10];
    &v[10..100];
    &v[10..];
    &v[..100];
}
