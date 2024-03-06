//@ compile-flags: -Znext-solver
//@ check-pass

// Not exactly sure if this is the inference behavior we *want*,
// but it is a side-effect of the lazy normalization of TAITs.

#![feature(type_alias_impl_trait)]

fn mk<T>() -> T {
    todo!()
}

fn main() {
    type Tait = impl Sized;
    type Tait2 = impl Sized;
    let x: Tait = 1u32;
    let y: Tait2 = x;
}
