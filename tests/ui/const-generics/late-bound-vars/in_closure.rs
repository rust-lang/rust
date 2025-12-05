//@ known-bug: unknown
// see comment on `tests/ui/const-generics/late-bound-vars/simple.rs`

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

const fn inner<'a>() -> usize
where
    &'a (): Sized,
{
    3
}

fn test<'a>() {
    let _ = || {
        let _: [u8; inner::<'a>()];
        let _ = [0; inner::<'a>()];
    };
}

fn main() {
    test();
}
