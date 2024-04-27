//@ run-pass
trait X {
    fn call<T: std::fmt::Debug>(&self, x: &T);
    fn default_method<T: std::fmt::Debug>(&self, x: &T) {
        println!("X::default_method {:?}", x);
    }
}

#[derive(Debug)]
struct Y(#[allow(dead_code)] isize);

#[derive(Debug)]
struct Z<T: X+std::fmt::Debug> {
    x: T
}

impl X for Y {
    fn call<T: std::fmt::Debug>(&self, x: &T) {
        println!("X::call {:?} {:?}", self, x);
    }
}

impl<T: X + std::fmt::Debug> Drop for Z<T> {
    fn drop(&mut self) {
        // These statements used to cause an ICE.
        self.x.call(self);
        self.x.default_method(self);
    }
}

pub fn main() {
    let _z = Z {x: Y(42)};
}
