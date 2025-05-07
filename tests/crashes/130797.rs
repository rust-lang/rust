//@ known-bug: #130797

trait Transform {
    type Output<'a>;
}
trait Propagate<O> {}
trait AddChild<C> {
    fn add_child(&self) {}
}

pub struct Node<T>(T);
impl<T> AddChild<Box<dyn for<'b> Propagate<T::Output<'b>>>> for Node<T> where T: Transform {}

fn make_graph_root() {
    Node(Dummy).add_child()
}

struct Dummy;
impl Transform for Dummy {
    type Output<'a> = ();
}

pub fn main() {}
