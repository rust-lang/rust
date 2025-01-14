//@ check-pass

// Regression test for #128176. Previously we would call `type_of` on the `1` anon const
// before the anon const had been lowered and had the `type_of` fed with a result.

#![feature(generic_const_exprs)]
#![feature(dyn_compatible_for_dispatch)]
#![allow(incomplete_features)]

trait X {
    type Y<const N: i16>;
}

const _: () = {
    fn f2<'a>(arg: Box<dyn X<Y<1> = &'a ()>>) {}
};

fn main() {}
