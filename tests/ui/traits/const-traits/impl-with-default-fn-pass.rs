//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

const trait Tr {
    fn req(&self);

    fn default() {}
}

const impl Tr for u8 {
    fn req(&self) {}
}

macro_rules! impl_tr {
    ($ty: ty) => {
        const impl Tr for $ty {
            fn req(&self) {}
        }
    };
}

impl_tr!(u64);

fn main() {}
