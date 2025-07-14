// FIXME(fmease): Description.
//@ check-pass

#![feature(return_type_notation)]

trait TraitA { type Assoc: TraitB; }
trait TraitB { fn assoc() -> impl Sized; }

fn scope<T: TraitA>()
where
    T::Assoc::assoc(..): Iterator<Item = u8>,
{
    let _: Vec<u8> = T::Assoc::assoc().collect();
}

fn main() {}
