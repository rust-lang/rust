// Regression test for #116519: incomplete impl with const param used to ICE in ArgFolder.

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait Ret {
    type R;
}

struct Cond<const PRED: bool, U, V>(std::marker::PhantomData<U>,);
//~^ ERROR type parameter `V` is never used

struct RobinHashTable<
    const MAX_LENGTH: usize,
    CellIdx = <Cond<{ }, u16, u32> as Ret>::R,
> {}
//~^^ ERROR type parameter `CellIdx` is never used

impl<CellIdx> HashMapBase<CellIdx> for RobinHashTable<MAX_LENGTH, CellIdx> {}
//~^ ERROR cannot find trait `HashMapBase` in this scope
//~| ERROR cannot find type `MAX_LENGTH` in this scope
//~| ERROR unresolved item provided when a constant was expected

fn main() {}
