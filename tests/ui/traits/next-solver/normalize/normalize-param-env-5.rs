//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#246
// Fixed by eager norm and marking param env as rigid.

struct MarkedStruct;
pub trait MarkerTrait {}
impl MarkerTrait for MarkedStruct {}

struct ChainStruct;
trait ChainTrait {}
impl ChainTrait for ChainStruct where MarkedStruct: MarkerTrait {}

pub struct FooStruct;
pub trait FooTrait {
    type Output;
}
pub struct FooOut;
impl FooTrait for FooStruct
where
    ChainStruct: ChainTrait,
{
    type Output = FooOut;
}
type FooOutAlias = <FooStruct as FooTrait>::Output;

pub trait Trait<T> {
    type Output;
}

pub fn foo<K>()
where
    FooOut: Trait<K>,
    <FooOutAlias as Trait<K>>::Output: MarkerTrait,
{
}

fn main() {}
