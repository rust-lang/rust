#![warn(clippy::indexing_slicing)]
// We also check the out_of_bounds_indexing lint here, because it lints similar things and
// we want to avoid false positives.
#![warn(clippy::out_of_bounds_indexing)]
#![allow(clippy::no_effect, clippy::unnecessary_operation)]

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    x[index];
    x[4]; // Ok, let rustc's `const_err` lint handle `usize` indexing on arrays.
    x[1 << 3]; // Ok, let rustc's `const_err` lint handle `usize` indexing on arrays.

    x[0]; // Ok, should not produce stderr.
    x[3]; // Ok, should not produce stderr.

    let y = &x;
    y[0]; // Ok, referencing shouldn't affect this lint. See the issue 6021
    y[4]; // Ok, rustc will handle references too.

    let v = vec![0; 5];
    v[0];
    v[10];
    v[1 << 3];

    const N: usize = 15; // Out of bounds
    const M: usize = 3; // In bounds
    x[N]; // Ok, let rustc's `const_err` lint handle `usize` indexing on arrays.
    x[M]; // Ok, should not produce stderr.
    v[N];
    v[M];
}
