// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn foo<const X: usize, const Y: usize>() -> usize {
    0
}

fn main() {
    foo::<0>();
    //~^ ERROR this function takes 2

    foo::<0, 0, 0>();
    //~^ ERROR this function takes 2
}
