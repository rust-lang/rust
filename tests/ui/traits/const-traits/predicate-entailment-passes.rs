//@ compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl, effects)]
//~^ WARN the feature `effects` is incomplete

#[const_trait] trait Bar {}
impl const Bar for () {}


#[const_trait] trait TildeConst {
    type Bar<T> where T: ~const Bar;

    fn foo<T>() where T: ~const Bar;
}
impl TildeConst for () {
    type Bar<T> = () where T: Bar;

    fn foo<T>() where T: Bar {}
}


#[const_trait] trait AlwaysConst {
    type Bar<T> where T: const Bar;

    fn foo<T>() where T: const Bar;
}
impl AlwaysConst for i32 {
    type Bar<T> = () where T: Bar;

    fn foo<T>() where T: Bar {}
}
impl const AlwaysConst for u32 {
    type Bar<T> = () where T: ~const Bar;

    fn foo<T>() where T: ~const Bar {}
}

fn main() {}
