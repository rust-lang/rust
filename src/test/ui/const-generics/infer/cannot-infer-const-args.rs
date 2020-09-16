// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn foo<const X: usize>() -> usize {
    0
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
