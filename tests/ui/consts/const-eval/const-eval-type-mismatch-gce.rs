//@ revisions: with-debug-assertions no-debug-assertions
// This test verifies ICE doesn't happen on mismatched types of the result of binary
// or unary operations during const eval. See issue #125772.
//@ [no-debug-assertions]   compile-flags: -Cdebug_assertions=no
//@ [with-debug-assertions] compile-flags: -Cdebug_assertions=yes
//@ only-64bit
#![feature(generic_const_exprs)]

struct Outer<const A: i64, const B: i64>();
impl<const A: usize, const B: usize> Outer<A, B>
//~^ ERROR the constant `A` is not of type `i64`
//~| ERROR the constant `B` is not of type `i64`
where
    [(); A + (B * 2)]:,
{
    fn i() -> Self {
        //~^ ERROR the constant `A` is not of type `i64`
        //~| ERROR the constant `B` is not of type `i64`
        Self
        //~^ ERROR mismatched types [E0308]
        //~| ERROR the constant `A` is not of type `i64`
        //~| ERROR the constant `B` is not of type `i64`
    }
}

fn main() {
    Outer::<1, 1>::o();
    //~^ ERROR no associated function or constant named `o` found for struct `Outer<A, B>` in the current scope [E0599]
}
