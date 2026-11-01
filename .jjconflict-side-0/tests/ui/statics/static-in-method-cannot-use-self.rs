//! Regression test for https://github.com/rust-lang/rust/issues/3668
//!
struct P {
    child: Option<Box<P>>,
}
trait PTrait {
    fn getChildOption(&self) -> Option<Box<P>>;
}

impl PTrait for P {
    fn getChildOption(&self) -> Option<Box<P>> {
        static childVal: Box<P> = self.child.get();
        //~^ ERROR attempt to use a non-constant value in a constant
        panic!();
    }
}

fn main() {}
