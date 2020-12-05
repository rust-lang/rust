#[derive(Copy, Clone)]
struct Flags;

trait A {
}

impl<T> Drop for T where T: A { //~ ERROR E0119
                                //~^ ERROR E0120
                                //~| ERROR E0210
    fn drop(&mut self) {
    }
}

fn main() {}
