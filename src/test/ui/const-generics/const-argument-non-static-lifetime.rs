// run-pass
// revisions: full
// FIXME(#75323) Omitted min revision for now due to ICE.

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
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
