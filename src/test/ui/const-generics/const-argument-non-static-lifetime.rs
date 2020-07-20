// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete
#![allow(dead_code)]

fn test<const N: usize>() {}

fn wow<'a>() -> &'a () {
    test::<{
        let _: &'a ();
        3
    }>();
    &()
}

fn main() {}
