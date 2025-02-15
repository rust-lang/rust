trait Sup {
    fn method(&self) {}
}

trait Trait: Sup {
    fn method(&self) {}
}

impl Sup for i32 {}
impl Trait for i32 {}

fn poly<T: Trait>(x: T) {
    x.method();
    //~^ ERROR multiple applicable items in scope
}

fn concrete() {
    1.method();
    //~^ ERROR multiple applicable items in scope
}

fn main() {}
