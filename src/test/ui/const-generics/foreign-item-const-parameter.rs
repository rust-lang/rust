// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

extern "C" {
    fn foo<const X: usize>(); //~ ERROR foreign items may not have const parameters

    fn bar<T, const X: usize>(_: T); //~ ERROR foreign items may not have type or const parameters
}

fn main() {}
