// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn foo<const X: usize>() -> usize {
    0
}

fn main() {
    foo(); //~ ERROR type annotations needed
}
