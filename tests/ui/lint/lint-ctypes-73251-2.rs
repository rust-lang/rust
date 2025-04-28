#![feature(type_alias_impl_trait)]
#![deny(improper_ctypes)]

pub trait TraitA {
    type Assoc;
}

impl TraitA for u32 {
    type Assoc = u32;
}

pub trait TraitB {
    type Assoc;
}

impl<T> TraitB for T
where
    T: TraitA,
{
    type Assoc = <T as TraitA>::Assoc;
}

type AliasA = impl TraitA<Assoc = u32>;

type AliasB = impl TraitB<Assoc = AliasA>;

#[define_opaque(AliasA)]
fn use_of_a() -> AliasA {
    3
}

#[define_opaque(AliasB)]
fn use_of_b() -> AliasB {
    3
}

extern "C" {
    fn lint_me() -> <AliasB as TraitB>::Assoc; //~ ERROR: uses type `AliasA`
}

fn main() {}
