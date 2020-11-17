// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn foo<const X: usize, const Y: usize>() -> usize {
    0
}

fn main() {
    foo::<0>(); //~ ERROR wrong number of const arguments: expected 2, found 1
    foo::<0, 0, 0>(); //~ ERROR wrong number of const arguments: expected 2, found 3
}
