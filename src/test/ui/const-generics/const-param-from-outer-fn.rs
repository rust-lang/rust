// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn foo<const X: u32>() {
    fn bar() -> u32 {
        X //~ ERROR can't use generic parameters from outer function
    }
}

fn main() {}
