use std::fmt::Debug;

struct D<T: HasArg>(T::Arg);

trait HasArg {
    type Arg: Debug;
}
impl<'a, T: Debug> HasArg for fn(&'a T) {
    type Arg = &'a T;
}
impl<T: HasArg> Drop for D<T> {
    fn drop(&mut self) {
        println!("{:?}", self.0);
    }
}
fn mk<'a, T: Debug>(r: &'a T) -> D<fn(&'a T)> {
    D(r)
}

fn main() {
    let b = Box::new(vec![vec![1]]);
    let d;
    d = mk(&*b);
    drop(b); //~ ERROR cannot move out of `b` because it is borrowed
}
