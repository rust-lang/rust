// can't use build-fail, because this also fails check-fail, but
// the ICE from #120787 only reproduces on build-fail.
//@ compile-flags: --emit=mir

#![feature(type_alias_impl_trait)]

struct Foo {
    field: String,
}

type Tait = impl Sized;

#[define_opaque(Tait)]
fn ice_cold(beverage: Tait) {
    let Foo { field } = beverage;
    _ = field;
}

fn main() {
    Ok(()) //~ ERROR mismatched types
}
