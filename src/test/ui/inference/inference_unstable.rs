// Ensures #[unstable] functions without opting in the corresponding #![feature]
// will not break inference.

// aux-build:inference_unstable_iterator.rs
// aux-build:inference_unstable_itertools.rs
// run-pass

extern crate inference_unstable_iterator;
extern crate inference_unstable_itertools;

#[allow(unused_imports)]
use inference_unstable_iterator::IpuIterator;
use inference_unstable_itertools::IpuItertools;

fn main() {
    assert_eq!('x'.ipu_flatten(), 1);
    //~^ WARN a method with this name may be added to the standard library in the future
    //~^^ WARN once this method is added to the standard library, the ambiguity may cause an error
}
