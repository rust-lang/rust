trait Foo {
    fn method(&self) {}
}

fn call_method<T: std::fmt::Debug>(x: &T) {
    x.method() //~ ERROR E0599
}

fn call_method_2<T>(x: T) {
    x.method() //~ ERROR E0599
}

fn main() {}
