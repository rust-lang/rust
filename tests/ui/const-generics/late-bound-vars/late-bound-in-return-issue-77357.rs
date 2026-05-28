//@ known-bug: unknown
// see comment on `tests/ui/const-generics/late-bound-vars/simple.rs`

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait MyTrait<T> {}

fn bug<'a, T>() -> &'static dyn MyTrait<[(); { |x: &'a u32| { x }; 4 }]> {
    todo!()
}

fn main() {}
