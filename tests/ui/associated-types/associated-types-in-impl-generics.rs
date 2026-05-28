//@ run-pass

trait Get {
    type Value;
    fn get(&self) -> &<Self as Get>::Value;
}

struct Struct {
    x: isize,
}

impl Get for Struct {
    type Value = isize;
    fn get(&self) -> &isize {
        &self.x
    }
}

trait Grab {
    type U;
    fn grab(&self) -> &<Self as Grab>::U;
}

impl<T:Get> Grab for T {
    type U = <T as Get>::Value;
    fn grab(&self) -> &<T as Get>::Value {
        self.get()
    }
}

fn main() {
    let s = Struct {
        x: 100,
    };
    assert_eq!(*s.grab(), 100);
}
