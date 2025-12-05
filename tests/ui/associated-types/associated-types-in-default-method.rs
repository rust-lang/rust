//@ run-pass

trait Get {
    type Value;
    fn get(&self) -> &<Self as Get>::Value;
    fn grab(&self) -> &<Self as Get>::Value {
        self.get()
    }
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

fn main() {
    let s = Struct {
        x: 100,
    };
    assert_eq!(*s.grab(), 100);
}
