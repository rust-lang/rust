// run-rustfix

#![warn(clippy::flat_map_identity)]

use std::convert;

fn main() {
    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    iterator.flat_map(|x| x);

    let iterator = [[0, 1], [2, 3], [4, 5]].iter();
    iterator.flat_map(convert::identity);
}
