//@ known-bug: #116519
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
trait Ret {
    type R;
}
struct Cond<const PRED: bool, U, V>(std::marker::PhantomData<U>, std::marker::PhantomData<V>);
impl<U, V> Ret for Cond<true, U, V> {
    type R = U;
}
impl<U, V> Ret for Cond<false, U, V> {
    type R = V;
}
struct RobinHashTable<
    const MAX_LENGTH: usize,
    CellIdx = <Cond<{ MAX_LENGTH < 65535 }, u16, u32> as Ret>::R,
> {
    _idx: CellIdx,
}
impl<CellIdx> RobinHashTable<MAX_LENGTH, CellIdx> {
    fn new() -> Self {
        Self {
            _idx: CellIdx { MAX_LENGTH },
        }
    }
}
impl<CellIdx> HashMapBase<CellIdx> {
    fn new() -> Self {
        Self {
            _idx: CellIdx { 0 },
        }
    }
}
impl<CellIdx> HashMapBase<CellIdx> for RobinHashTable<MAX_LENGTH, CellIdx> {
    fn hash<H: Hash + Hasher>(&self,

    ) -> H {
        self._idx.hash()
    }
    fn eq(&self, other: &Self) -> bool {
        self._idx.eq(other._idx)
    }
}
impl<CellIdx> HashMapBase<CellIdx> for RobinHashTable<MAX_LENGTH, CellIdx> {
    fn hash<H: Hash + Hasher>(&self, other: &Self) -> H {
        self._idx.hash(other._idx)
    }
    fn eq(&self, other: &Self) -> bool {
        self._idx.eq(other._idx)
    }
}
#[test]
fn test_size_of_robin_hash_table() {
    use std::mem::size_of;
    println!("{}", size_of::<RobinHashTable<1024>>());
    println!("{}", size_of::<RobinHashTable<65536>>());
}
