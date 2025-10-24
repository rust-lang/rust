//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for trait-system-refactor-initiative#246.

struct MarkedStruct;
pub trait MarkerTrait {}
impl MarkerTrait for MarkedStruct {}

// Necessary indirection to get a cycle with `PathKind::Unknown`
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

pub trait Trait<T> {
    type Output;
}

// prove <<FooStruct as FooTrait>::Output as Trait<K>>::Output: MarkerTrait
// normalize <<FooStruct as FooTrait>::Output as Trait<K>>::Output
//   `<?fresh_infer as Trait<K>>::Output` remains ambiguous, so this alias is never rigid
// alias-relate <FooStruct as FooTrait>::Output ?fresh_infer
//   does not constrain `?fresh_infer` as `keep_constraints` is `false`
// normalize <FooStruct as FooTrait>::Output
//   result `FooOut` with overflow
// prove ChainStruct: ChainTrait
// prove MarkedStruct: MarkerTrait
//   impl trivial (ignored)
//   where-clause requires normalize <<FooStruct as FooTrait>::Output as Trait<K>>::Output overflow
pub fn foo<K>()
where
    FooOut: Trait<K>,
    <<FooStruct as FooTrait>::Output as Trait<K>>::Output: MarkerTrait,
{
}

fn main() {}
