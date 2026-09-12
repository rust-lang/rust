//@ compile-flags: -Znext-solver=globally
//@ edition: 2021
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/156345>.
// An `extern "C" fn` taking a type alias impl trait argument used to ICE with
// the new solver, leaving an entry in the `OpaqueTypeStorage`. Only the new
// solver was affected.

#![feature(type_alias_impl_trait)]
#![allow(improper_ctypes_definitions)]

struct Foo {
    field: String,
}

type Tait = impl Sized;

#[define_opaque(Tait)]
extern "C" fn ice_cold(beverage: Tait) {
    let Foo { field } = beverage;
    let _ = field;
}

// A second reproducer from the same issue, with the opaque type in return
// position behind a higher-ranked closure bound.
struct Parser<H>(H);

impl<H, T> Parser<H>
where
    H: for<'a> Fn(&'a str) -> T,
{
    fn new(handler: H) -> Parser<H> {
        Parser(handler)
    }

    extern "C" fn many<'s>() -> Parser<impl for<'a> Fn(&'a str) + 's> {
        Parser::new(|_| ())
    }
}

fn main() {}
