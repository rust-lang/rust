//@ known-bug: rust-lang/rust#128176

#![feature(generic_const_exprs)]
#![feature(dyn_compatible_for_dispatch)]
trait X {
    type Y<const N: i16>;
}

const _: () = {
    fn f2<'a>(arg: Box<dyn X<Y<1> = &'a ()>>) {}
};

fn main() {}
