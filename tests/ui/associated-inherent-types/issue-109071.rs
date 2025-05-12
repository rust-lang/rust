//@ revisions: with_gate no_gate
#![cfg_attr(with_gate, feature(inherent_associated_types))]
#![cfg_attr(with_gate, allow(incomplete_features))]

struct Windows<T> { t: T }

impl<T> Windows { //~ ERROR: missing generics for struct `Windows`
    type Item = &[T]; //~ ERROR: `&` without an explicit lifetime name cannot be used here
    //[no_gate]~^ ERROR: inherent associated types are unstable

    fn next() -> Option<Self::Item> {}
}

impl<T> Windows<T> {
    fn T() -> Option<Self::Item> {}
    //[no_gate]~^ ERROR: ambiguous associated type
}

fn main() {}
