#![allow(unused_imports, clippy::needless_return)]
#![warn(clippy::flat_map_identity)]

use std::convert;

fn main() {
    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(|x| x);

    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(convert::identity);

    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    let _ = iterator.flat_map(|x| return x);
}
