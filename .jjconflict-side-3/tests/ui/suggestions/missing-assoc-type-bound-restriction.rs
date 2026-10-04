//@ check-pass

trait Parent {
    type Ty;
    type Assoc: Child<Self::Ty>;
}

trait Child<T> {}

struct ChildWrapper<T>(T);

impl<A, T> Child<A> for ChildWrapper<T> where T: Child<A> {}

struct ParentWrapper<T>(T);

impl<A, T: Parent<Ty = A>> Parent for ParentWrapper<T> {
    type Ty = A;
    type Assoc = ChildWrapper<T::Assoc>;
}

fn main() {}
