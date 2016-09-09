struct Struct(i32);

trait Trait {
    fn method(&self);
}

impl Trait for Struct {
    fn method(&self) {
        assert_eq!(self.0, 42);
    }
}

fn main() {
    let y: &Trait = &Struct(42);
    y.method();
}
