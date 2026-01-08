#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

pub trait IsVoid {
    #[type_const]
    const IS_VOID: bool;
}
impl IsVoid for () {
    #[type_const]
    const IS_VOID: bool = true;
}

pub trait Maybe {}
impl Maybe for () {}
impl Maybe for () where (): IsVoid<IS_VOID = true> {}
//~^ ERROR conflicting implementations of trait `Maybe` for type `()`

fn main() {}
