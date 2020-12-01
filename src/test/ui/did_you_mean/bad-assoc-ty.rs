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
//~| ERROR the type placeholder `_` is not allowed within types on item signatures

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
//~^ ERROR ambiguous associated type

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
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

fn bar<F>(_: F) where F: Fn() -> _ {}
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

fn baz<F: Fn() -> _>(_: F) {}
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

struct L<F>(F) where F: Fn() -> _;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
struct M<F> where F: Fn() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    a: F,
}
enum N<F> where F: Fn() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    Foo(F),
}

union O<F> where F: Fn() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
    foo: F,
}

trait P<F> where F: Fn() -> _ {
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures
}

trait Q {
    fn foo<F>(_: F) where F: Fn() -> _ {}
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
}

fn main() {}
