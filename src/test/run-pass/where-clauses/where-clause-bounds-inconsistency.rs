// run-pass
// pretty-expanded FIXME #23616

trait Bound {
    fn dummy(&self) { }
}

trait Trait {
    fn a<T>(&self, _: T) where T: Bound;
    fn b<T>(&self, _: T) where T: Bound;
    fn c<T: Bound>(&self, _: T);
    fn d<T: Bound>(&self, _: T);
}

impl Trait for bool {
    fn a<T: Bound>(&self, _: T) {}
    //^~ This gets rejected but should be accepted
    fn b<T>(&self, _: T) where T: Bound {}
    fn c<T: Bound>(&self, _: T) {}
    fn d<T>(&self, _: T) where T: Bound {}
}

fn main() {}
