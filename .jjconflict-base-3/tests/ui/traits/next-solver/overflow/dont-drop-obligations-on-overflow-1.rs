//@ compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#294
// We used to drop all subsequent obligations when one obligation overflows
// in fulfillment. It means we don't really prove all obligations even if
// fulfillment returns no error.
//
// We now eagerly abort on the first overflowed obligation.

#![feature(impl_trait_in_assoc_type)]
#![forbid(unsafe_code)]

trait Amb<'z> {}
trait Sub<'c, 'd>: Amb<'c> + Amb<'d> {}
impl<'z> Amb<'z> for i32 {}
impl<'c, 'd> Sub<'c, 'd> for i32 {}

trait Call<'a> {
    type Output;
    fn call() -> Self::Output;
}

trait Leak<G> {
    fn leak(self) -> &'static u8;
}
impl<'z, G: Call<'static, Output = R>, R: Amb<'z>> Leak<G> for &'static u8 {
    fn leak(self) -> &'static u8 {
        self
    }
}

#[expect(dead_code)]
struct Foo<'c, 'd>(&'c (), &'d ());

impl<'a, 'c, 'd> Call<'a> for Foo<'c, 'd>
where
    i32: Sub<'c, 'd>,
{
    type Output = impl Sized + use<>;
    fn call() -> Self::Output {
        let r = {
            let local = 42_u8;
            <&u8 as Leak<Foo<'c, 'd>>>::leak(&local)
            //~^ ERROR: overflow evaluating the requirement `&u8: Leak<Foo<'c, 'd>>`
        };
        println!("{r}"); // use-after-free
        1_i32
    }
}

fn main() {
    Foo::call();
}
