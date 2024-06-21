type A = rustfmt; //~ ERROR expected type, found tool module `rustfmt`
type B = rustfmt::skip; //~ ERROR expected type, found tool attribute `rustfmt::skip`

#[derive(rustfmt)] //~ ERROR cannot find derive macro `rustfmt`
                   //~| ERROR cannot find derive macro `rustfmt`
struct S;

// Interpreted as an unstable custom attribute
#[rustfmt] //~ ERROR cannot find attribute `rustfmt`
fn check() {}

#[rustfmt::skip] // OK
fn main() {
    rustfmt; //~ ERROR expected value, found tool module `rustfmt`
    rustfmt!(); //~ ERROR cannot find macro `rustfmt`

    rustfmt::skip; //~ ERROR expected value, found tool attribute `rustfmt::skip`
}
