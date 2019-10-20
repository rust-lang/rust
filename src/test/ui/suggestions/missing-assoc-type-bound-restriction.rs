// Running rustfix would cause the same suggestion to be applied multiple times, which results in
// invalid code.

trait Parent {
    type Ty;
    type Assoc: Child<Self::Ty>;
}

trait Child<T> {}

struct ChildWrapper<T>(T);

impl<A, T> Child<A> for ChildWrapper<T> where T: Child<A> {}

struct ParentWrapper<T>(T);

impl<A, T: Parent<Ty = A>> Parent for ParentWrapper<T> {
    //~^ ERROR the trait bound `<T as Parent>::Assoc: Child<A>` is not satisfied
    //~| ERROR the trait bound `<T as Parent>::Assoc: Child<A>` is not satisfied
    type Ty = A;
    type Assoc = ChildWrapper<T::Assoc>;
    //~^ ERROR the trait bound `<T as Parent>::Assoc: Child<A>` is not satisfied
}

fn main() {}
