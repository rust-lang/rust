// FIXME(fmease): Description.
//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait TraitA { type Assoc: TraitB; }
trait TraitB { #[type_const] const ASSOC: usize; }

fn scope<T: TraitA>() -> [u8; T::Assoc::ASSOC] {
    [0; T::Assoc::ASSOC]
}

fn main() {}
