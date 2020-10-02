// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

fn foo<const X: u32>() {
    fn bar() -> u32 {
        X //~ ERROR can't use generic parameters from outer function
    }
}

fn main() {}
