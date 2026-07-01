type A = rustfmt; //~ ERROR cannot find type `rustfmt`
type B = rustfmt::skip; //~ ERROR cannot find module or crate `rustfmt`

#[derive(rustfmt)] //~ ERROR cannot find derive macro `rustfmt` in this scope
                   //~| ERROR cannot find derive macro `rustfmt` in this scope
struct S;

// Interpreted as an unstable custom attribute
#[rustfmt] //~ ERROR cannot find attribute `rustfmt` in this scope
fn check() {}

#[rustfmt::skip] // OK
fn main() {
    rustfmt; //~ ERROR cannot find value `rustfmt`
    rustfmt!(); //~ ERROR cannot find macro `rustfmt` in this scope

    rustfmt::skip; //~ ERROR cannot find module or crate `rustfmt`
}
