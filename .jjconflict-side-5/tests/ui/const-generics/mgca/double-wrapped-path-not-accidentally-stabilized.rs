//! When doing some refactorings in mGCA, it was easy to accidentally stabilize doubly-brace-wrapped
//! paths. Check to make sure we don't accidentally do so.
//!
//! Feel free to delete this test if/when mGCA is stabilized and we support this syntax on stable,
//! it's testing nothing useful beyond that point.

fn f<const N: usize>() {}

fn g<const N: usize>() {
    f::<{ N }>(); // ok
    f::<{ { N } }>();
    //~^ ERROR: generic parameters may not be used in const operations
}

fn main() {}
