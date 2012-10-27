trait Mumbo {
    pure fn jumbo(&self, x: @uint) -> uint;
    fn jambo(&self, x: @const uint) -> uint;
    fn jbmbo(&self) -> @uint;
}

impl uint: Mumbo {
    // Cannot have a larger effect than the trait:
    fn jumbo(&self, x: @uint) { *self + *x; }
    //~^ ERROR expected pure fn but found impure fn

    // Cannot accept a narrower range of parameters:
    fn jambo(&self, x: @uint) { *self + *x; }
    //~^ ERROR values differ in mutability

    // Cannot return a wider range of values:
    fn jbmbo(&self) -> @const uint { @const 0 }
    //~^ ERROR values differ in mutability
}

fn main() {}




