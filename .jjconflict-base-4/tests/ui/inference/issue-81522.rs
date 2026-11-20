// Regression test for #81522.
// Ensures that `#[allow(unstable_name_collisions)]` appended to things other than function
// suppresses the corresponding diagnostics emitted from inside them.
// But note that this attribute doesn't work for macro invocations if it is appended directly.

//@ aux-build:inference_unstable_iterator.rs
//@ aux-build:inference_unstable_itertools.rs
//@ run-pass

extern crate inference_unstable_iterator;
extern crate inference_unstable_itertools;

#[allow(unused_imports)]
use inference_unstable_iterator::IpuIterator;
use inference_unstable_itertools::IpuItertools;

fn main() {
    // expression statement
    #[allow(unstable_name_collisions)]
    'x'.ipu_flatten();

    // let statement
    #[allow(unstable_name_collisions)]
    let _ = 'x'.ipu_flatten();

    // block expression
    #[allow(unstable_name_collisions)]
    {
        'x'.ipu_flatten();
    }
}
