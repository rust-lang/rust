use std::ops::Deref;

trait PointerFamily {
    type Pointer<T>: Deref<Target = T>;
}

struct RcFamily;

impl PointerFamily for RcFamily {
    type Pointer<T> = dyn Deref<Target = T>;
    //~^ ERROR the size for values of type `(dyn Deref<Target = T> + 'static)` cannot be known at compilation time
}

enum Node<T, P: PointerFamily> {
    Cons(T, P::Pointer<Node<T, P>>),
    Nil,
}

type RcNode<T> = Node<T, RcFamily>;

impl<T, P: PointerFamily> Node<T, P>
where
    P::Pointer<Node<T, P>>: Sized,
{
    fn new() -> P::Pointer<Self> {
        todo!()
    }
}

fn main() {
    let mut list = RcNode::<i32>::new();
    //~^ ERROR the variant or associated item `new` exists for enum `Node<i32, RcFamily>`, but its trait bounds were not satisfied
}
