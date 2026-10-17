//@ known-bug: #130524

trait Transform {
    type Output<'a>;
}

trait Propagate<Input> {}

fn new_node<T: Transform>(_c: Vec<Box<dyn for<'a> Propagate<<T as Transform>::Output<'a>>>>) -> T {
    todo!()
}

impl<Input, T> Propagate<Input> for T {}
struct Noop;

impl Transform for Noop {
    type Output<'a> = ();
}

fn main() {
    let _node: Noop = new_node(vec![Box::new(Noop)]);
}
