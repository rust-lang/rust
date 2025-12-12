//@ check-pass

// We allow for literals to implicitly be anon consts still regardless
// of whether a const block is placed around them or not

#![feature(min_generic_const_args, associated_const_equality)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const ASSOC: isize;
}

fn ace<T: Trait<ASSOC = 1, ASSOC = -1>>() {}
fn repeat_count() {
    [(); 1];
}
type ArrLen = [(); 1];
struct Foo<const N: isize>;
type NormalArg = (Foo<1>, Foo<-1>);

fn main() {}
