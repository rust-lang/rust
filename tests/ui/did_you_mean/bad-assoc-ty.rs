//@revisions: edition2015 edition2021
//@[edition2015] edition:2015
//@[edition2021] edition:2021

type A = [u8; 4]::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type B = [u8]::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type C = (u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type D = (u8, u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

type E = _::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR the placeholder `_` is not allowed within types on item signatures for type aliases

type F = &'static (u8)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

// Qualified paths cannot appear in bounds, so the recovery
// should apply to the whole sum and not `(Send)`.
type G = dyn 'static + (Send)::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

// This is actually a legal path with fn-like generic arguments in the middle!
// Recovery should not apply in this context.
type H = Fn(u8) -> (u8)::Output;
//[edition2015]~^ ERROR ambiguous associated type
//[edition2015]~| WARN trait objects without an explicit `dyn` are deprecated
//[edition2015]~| WARN this is accepted in the current edition
//[edition2021]~^^^^ ERROR expected a type, found a trait

macro_rules! ty {
    ($ty: ty) => ($ty::AssocTy);
    //~^ ERROR missing angle brackets in associated item path
    //~| ERROR ambiguous associated type
    () => (u8);
}

type J = ty!(u8);
type I = ty!()::AssocTy;
//~^ ERROR missing angle brackets in associated item path
//~| ERROR ambiguous associated type

trait K<A, B> {}
fn foo<X: K<_, _>>(x: X) {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
//~| ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn bar<F>(_: F) where F: Fn() -> _ {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

fn baz<F: Fn() -> _>(_: F) {}
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions

struct L<F>(F) where F: Fn() -> _;
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for structs
struct M<F> where F: Fn() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for structs
    a: F,
}
enum N<F> where F: Fn() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for enums
    Foo(F),
}

union O<F> where F: Fn() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for unions
    foo: F,
    //~^ ERROR must implement `Copy`
}

trait P<F> where F: Fn() -> _ {
//~^ ERROR the placeholder `_` is not allowed within types on item signatures for traits
}

trait Q {
    fn foo<F>(_: F) where F: Fn() -> _ {}
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for functions
}

fn main() {}
