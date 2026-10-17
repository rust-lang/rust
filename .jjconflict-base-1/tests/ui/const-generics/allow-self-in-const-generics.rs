// Allow Self in const generics when Self doesn't depends on generics(#149203)
#![feature(min_adt_const_params)]

//1
trait MyTrait {
    fn foo<const N: i32>();
}

impl MyTrait for i32 {
    fn foo<const N: Self>() {}
}

//2
impl<T> Wrap<T> {
    fn f<const N: Self>() {}
    //~^ ERROR the type of const parameters must not depend on other generic parameters

}
struct Wrap<T>(T);

//3
type Foo<const N: usize> = Bar;

#[derive(Eq, PartialEq, core::marker::ConstParamTy)]
struct Bar;

trait Trait<const N: usize> {
    fn bar<const C: Bar>();
}

impl<const N: usize> Trait<N> for Foo<N> {
    fn bar<const C: Self>() {}
    // FIXME: currently the compiler let this pass
    // https://github.com/rust-lang/rust/pull/157949#discussion_r3544858218
}
fn main(){}
