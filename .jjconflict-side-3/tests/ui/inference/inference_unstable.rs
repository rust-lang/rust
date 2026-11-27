// Ensures #[unstable] functions without opting in the corresponding #![feature]
// will not break inference.

//@ aux-build:inference_unstable_iterator.rs
//@ aux-build:inference_unstable_itertools.rs
//@ run-pass

extern crate inference_unstable_iterator;
extern crate inference_unstable_itertools;

#[allow(unused_imports)]
use inference_unstable_iterator::IpuIterator;
use inference_unstable_itertools::IpuItertools;

fn main() {
    assert_eq!('x'.ipu_flatten(), 1);
//~^ WARN a method with this name may be added to the standard library in the future
//~| WARN once this associated item is added to the standard library, the ambiguity may cause an
    assert_eq!('x'.ipu_by_value_vs_by_ref(), 1);
//~^ WARN a method with this name may be added to the standard library in the future
//~| WARN once this associated item is added to the standard library, the ambiguity may cause an
    assert_eq!('x'.ipu_by_ref_vs_by_ref_mut(), 1);
//~^ WARN a method with this name may be added to the standard library in the future
//~| WARN once this associated item is added to the standard library, the ambiguity may cause an
    assert_eq!((&mut 'x' as *mut char).ipu_by_mut_ptr_vs_by_const_ptr(), 1);
//~^ WARN a method with this name may be added to the standard library in the future
//~| WARN once this associated item is added to the standard library, the ambiguity may cause an
    assert_eq!(char::C, 1);
//~^ WARN an associated constant with this name may be added to the standard library in the future
//~| WARN once this associated item is added to the standard library, the ambiguity may cause an
}
