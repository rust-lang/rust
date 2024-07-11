//@ known-bug: rust-lang/rust#126944
// Step 1: Create two names for a single type: `Thing` and `AlsoThing`

struct Thing;
struct Dummy;
pub trait DummyTrait {
    type DummyType;
}
impl DummyTrait for Dummy {
    type DummyType = Thing;
}
type AlsoThing = <Dummy as DummyTrait>::DummyType;

// Step 2: Create names for a single trait object type: `TraitObject` and `AlsoTraitObject`

pub trait SomeTrait {
    type Item;
}
type TraitObject = dyn SomeTrait<Item = AlsoThing>;
type AlsoTraitObject = dyn SomeTrait<Item = Thing>;

// Step 3: Force the compiler to check whether the two names are the same type

pub trait Supertrait {
    type Foo;
}
pub trait Subtrait: Supertrait<Foo = TraitObject> {}

pub trait HasOutput<A: ?Sized> {
    type Output;
}

fn foo<F>() -> F::Output
where
    F: HasOutput<dyn Subtrait<Foo = AlsoTraitObject>>,
{
    todo!()
}
