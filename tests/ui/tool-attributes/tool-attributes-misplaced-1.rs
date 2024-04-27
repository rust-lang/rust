type A = rustfmt; //~ ERROR expected type, found tool module `rustfmt`
type B = rustfmt::skip; //~ ERROR expected type, found tool attribute `rustfmt::skip`

#[derive(rustfmt)] //~ ERROR cannot find derive macro `rustfmt` in this scope
                   //~| ERROR cannot find derive macro `rustfmt` in this scope
struct S;

// Interpreted as an unstable custom attribute
#[rustfmt] //~ ERROR cannot find attribute `rustfmt` in this scope
fn check() {}

#[rustfmt::skip] // OK
fn main() {
    rustfmt; //~ ERROR expected value, found tool module `rustfmt`
    rustfmt!(); //~ ERROR cannot find macro `rustfmt` in this scope

    rustfmt::skip; //~ ERROR expected value, found tool attribute `rustfmt::skip`
}
