// Second case reported in issue #108894.

use std::marker::PhantomData;

#[derive(PartialEq, Eq)]
pub struct Id<T>(PhantomData<T>);

// manual implementation which would break the usage of const patterns
// impl<T> PartialEq for Id<T> { fn eq(&self, _: &Id<T>) -> bool { true } }
// impl<T> Eq for Id<T> {}

// This derive is undesired but cannot be removed without
// breaking the usages below
// #[derive(PartialEq, Eq)]
struct SomeNode();

fn accept_eq(_: &impl PartialEq) { }

fn main() {
    let node = Id::<SomeNode>(PhantomData);

    // this will only work if
    // - `Partial/Eq` is implemented manually, or
    // - `SomeNode` also needlessly(?) implements `Partial/Eq`
    accept_eq(&node); //~ ERROR can't compare `SomeNode` with `SomeNode`

    const CONST_ID: Id::<SomeNode> = Id::<SomeNode>(PhantomData);
    // this will work only when `Partial/Eq` is being derived
    // otherwise: error: to use a constant of type `Id<SomeNode>` in a pattern,
    //   `Id<SomeNode>` must be annotated with `#[derive(PartialEq, Eq)]`
    match node {
        CONST_ID => {}
    }
}
