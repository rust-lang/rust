// Test that the "imperfect derives" note is emitted for associated types,
// `Rc<T>`, and coinductive types.

use std::rc::Rc;

trait Trait {
    type Assoc: Clone;
}

#[derive(Clone)]
struct AssocStruct<T: Trait> {
    field: T::Assoc,
}

struct NonClone;
impl Trait for NonClone {
    type Assoc = u32;
}

#[derive(Clone)]
struct RcStruct<T> {
    field: Rc<T>,
}

#[derive(Clone)]
struct List<T> {
    value: Rc<T>,
    next: Option<Box<List<T>>>,
}

fn require_clone<T: Clone>() {}

fn main() {
    require_clone::<AssocStruct<NonClone>>();
    //~^ ERROR the trait bound `NonClone: Clone` is not satisfied

    require_clone::<RcStruct<NonClone>>();
    //~^ ERROR the trait bound `NonClone: Clone` is not satisfied

    require_clone::<List<NonClone>>();
    //~^ ERROR the trait bound `NonClone: Clone` is not satisfied
}
