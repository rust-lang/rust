// Extending Num and using inherited static methods

use num::from_int;

trait Num {
    static fn from_int(i: int) -> self;
    fn gt(&self, other: &self) -> bool;
}

pub trait NumExt: Num { }

fn greater_than_one<T:NumExt>(n: &T) -> bool {
    n.gt(&from_int(1))
}

fn main() {}
