//@ run-pass
fn main() {
    let x = X(15);
    let y = x.foo();
    println!("{:?}",y);
}

trait Foo
    where for<'a> &'a Self: Bar
{
    fn foo<'a>(&'a self) -> <&'a Self as Bar>::Output;
}

trait Bar {
    type Output;
}

struct X(i32);

impl<'a> Bar for &'a X {
    type Output = &'a i32;
}

impl Foo for X {
    fn foo<'a>(&'a self) -> <&'a Self as Bar>::Output {
        &self.0
    }
}
