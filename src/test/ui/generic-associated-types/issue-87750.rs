#![feature(generic_associated_types)]

trait PointerFamily {
    type Pointer<T>;
}

struct Rc<T>(Box<T>);
struct RcFamily;

impl PointerFamily for RcFamily {
    type Pointer<T> = Rc<T>;
}

#[allow(dead_code)]
enum Node<T, P: PointerFamily> where P::Pointer<Node<T, P>>: Sized {
    Cons(P::Pointer<Node<T, P>>),
}

fn main() {
    let _list: <RcFamily as PointerFamily>::Pointer<Node<i32, RcFamily>>;
    //~^ ERROR overflow evaluating the requirement `Node<i32, RcFamily>: Sized`
}
