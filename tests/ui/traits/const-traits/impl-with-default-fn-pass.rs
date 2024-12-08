//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait Tr {
    fn req(&self);

    fn default() {}
}

impl const Tr for u8 {
    fn req(&self) {}
}

macro_rules! impl_tr {
    ($ty: ty) => {
        impl const Tr for $ty {
            fn req(&self) {}
        }
    }
}

impl_tr!(u64);

fn main() {}
