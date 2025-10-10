#![allow(unused_imports, clippy::needless_return)]
#![warn(clippy::flat_map_identity)]

use std::convert;

fn main() {
    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(|x| x);
    //~^ flat_map_identity

    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(convert::identity);
    //~^ flat_map_identity

    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(|x| return x);
    //~^ flat_map_identity
}

fn issue15198() {
    let x = [[1, 2], [3, 4]];
    // don't lint: this is an `Iterator<Item = &[i32, i32]>`
    // match ergonomics makes the binding patterns into references
    // so that its type changes to `Iterator<Item = [&i32, &i32]>`
    let _ = x.iter().flat_map(|[x, y]| [x, y]);
    let _ = x.iter().flat_map(|x| [x[0]]);

    // no match ergonomics for `[i32, i32]`
    let _ = x.iter().copied().flat_map(|[x, y]| [x, y]);
    //~^ flat_map_identity
}
