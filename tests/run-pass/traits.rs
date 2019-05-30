struct Struct(i32);

trait Trait {
    fn method(&self);
}

impl Trait for Struct {
    fn method(&self) {
        assert_eq!(self.0, 42);
    }
}

struct Foo<T: ?Sized>(T);

fn main() {
    let y: &dyn Trait = &Struct(42);
    y.method();
    let x: Foo<Struct> = Foo(Struct(42));
    let y: &Foo<dyn Trait> = &x;
    y.0.method();

    let x: Box<dyn Fn(i32) -> i32> = Box::new(|x| x * 2);
    assert_eq!(x(21), 42);
    let mut i = 5;
    {
        let mut x: Box<dyn FnMut()> = Box::new(|| i *= 2);
        x(); x();
    }
    assert_eq!(i, 20);
}
