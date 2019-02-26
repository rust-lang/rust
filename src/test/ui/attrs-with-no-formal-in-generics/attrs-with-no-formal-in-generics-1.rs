// This test checks variations on `<#[attr] 'a, #[oops]>`, where
// `#[oops]` is left dangling (that is, it is unattached, with no
// formal binding following it).

#![feature(rustc_attrs)]

struct RefIntPair<'a, 'b>(&'a u32, &'b u32);

impl<#[rustc_1] 'a, 'b, #[oops]> RefIntPair<'a, 'b> {
    //~^ ERROR trailing attribute after generic parameter
}

fn main() {

}
