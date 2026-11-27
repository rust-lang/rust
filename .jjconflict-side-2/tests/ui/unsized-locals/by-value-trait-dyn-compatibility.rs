pub trait Foo {
    fn foo(self) -> String
    where
        Self: Sized;
}

struct A;

impl Foo for A {
    fn foo(self) -> String {
        format!("hello")
    }
}

fn main() {
    let x = *(Box::new(A) as Box<dyn Foo>); //~ERROR the size for values of type `dyn Foo` cannot be known at compilation time [E0277]
    x.foo();
    //~^ERROR the `foo` method cannot be invoked on a trait object
}
