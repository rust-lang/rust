#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::raw_handle;

fn main() {
    let mut arr = [1, 2, 3];
    let _ = raw_handle!(arr[0]);
    //~^ ERROR: cannot take address of a temporary [E0745]
    arr[0] = 2;
}
