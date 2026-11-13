// Regression test for ICE from issue #153366.

#![feature(unboxed_closures)]

fn iso<A>(a: Fn) -> Option<_>
//~^ ERROR missing generics for trait `Fn`
//~| ERROR the placeholder `_` is not allowed within types on item signatures for return types
//~| WARN trait objects without an explicit `dyn` are deprecated
//~| WARN this is accepted in the current edition
where
    dyn Fn(A) -> (): Sized,
{
    Box::new(iso_un_option)
    //~^ ERROR mismatched types
}
fn iso_un_option<B>() -> Box<_> {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    iso(())
    //~^ ERROR the size for values of type `(dyn Fn(_) + 'static)` cannot be known at compilation
}

fn main() {
    iso(())
    //~^ ERROR the size for values of type `(dyn Fn(_) + 'static)` cannot be known at compilation
}
