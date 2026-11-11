#![warn(clippy::indexing_slicing)]
#![allow(
    unconditional_panic,
    clippy::needless_lifetimes,
    clippy::no_effect,
    // This lint should be enabled again after issue
    // <https://github.com/rust-lang/rust-clippy/issues/17117> is fixed and `clippy::out_of_bounds_indexing`
    // and `unconditional_panic` don't overlap anymore.
    // We want to check for `clippy::out_of_bounds_indexing` because it lints similar things and
    // we want to avoid false positives.
    clippy::out_of_bounds_indexing,
    clippy::unnecessary_operation,
    clippy::useless_vec
)]

const ARR: [i32; 2] = [1, 2];
const REF: &i32 = &ARR[idx()]; // Ok, should not produce stderr, since `suppress-restriction-lint-in-const` is set true.

const fn idx() -> usize {
    1
}
const fn idx4() -> usize {
    4
}

fn main() {
    let x = [1, 2, 3, 4];
    let index: usize = 1;
    x[index];
    //~^ indexing_slicing
    x[4]; // Ok, let rustc's `unconditional_panic` lint handle `usize` indexing on arrays.
    x[1 << 3]; // Ok, let rustc's `unconditional_panic` lint handle `usize` indexing on arrays.

    x[0]; // Ok, should not produce stderr.
    x[3]; // Ok, should not produce stderr.
    x[const { idx() }]; // Ok, should not produce stderr.
    x[const { idx4() }]; // Ok, let rustc's `unconditional_panic` lint handle `usize` indexing on arrays.
    const { &ARR[idx()] }; // Ok, should not produce stderr, since `suppress-restriction-lint-in-const` is set true.

    let y = &x;
    y[0]; // Ok, referencing shouldn't affect this lint. See the issue 6021
    y[4]; // Ok, rustc will handle references too.

    let v = vec![0; 5];
    v[0];
    //~^ indexing_slicing
    v[10];
    //~^ indexing_slicing
    v[1 << 3];
    //~^ indexing_slicing

    const N: usize = 15; // Out of bounds
    const M: usize = 3; // In bounds
    x[N]; // Ok, let rustc's `unconditional_panic` lint handle `usize` indexing on arrays.
    x[M]; // Ok, should not produce stderr.
    v[N];
    //~^ indexing_slicing
    v[M];
    //~^ indexing_slicing
}

/// An opaque integer representation
pub struct Integer<'a> {
    /// The underlying data
    value: &'a [u8],
}
impl<'a> Integer<'a> {
    // Check whether `self` holds a negative number or not
    pub const fn is_negative(&self) -> bool {
        self.value[0] & 0b1000_0000 != 0
    }
}
