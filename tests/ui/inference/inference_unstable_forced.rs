// If the unstable API is the only possible solution,
// still emit E0658 "use of unstable library feature".

// aux-build:inference_unstable_iterator.rs

extern crate inference_unstable_iterator;

use inference_unstable_iterator::IpuIterator;

fn main() {
    assert_eq!('x'.ipu_flatten(), 0);   //~ ERROR E0658
}
