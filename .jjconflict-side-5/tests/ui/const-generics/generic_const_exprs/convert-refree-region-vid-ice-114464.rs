// ICE cannot convert Refree.. to a region vid
// issue: rust-lang/rust#114464

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn test<const N: usize>() {}

fn wow<'a>() {
    test::<{
        let _: &'a ();
        //~^ ERROR cannot capture late-bound lifetime in constant
        3
    }>();
}

fn main() {}
