#![feature(type_alias_impl_trait)]

type Tait = impl Copy;
// Make sure that this TAIT isn't considered unconstrained...

#[define_opaque(Tait)]
fn empty_opaque() -> Tait {
    if false {
        match empty_opaque() {}
        //~^ ERROR non-empty
    }
    0u8
}

fn main() {}
