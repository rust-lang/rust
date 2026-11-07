// #41966
trait Foo {}

struct Bar<R>(R);

impl<R> Foo for Bar<R> {
}

fn bb<R>(r: R) -> Box<dyn Foo> {
    Box::new(Bar(r)) //~ ERROR the parameter type `R` may not live long enough
}

fn cc<R>(r: R) -> Box<dyn Foo + '_> { //~ ERROR missing lifetime specifier
    Box::new(Bar(r))
}

// #54753
pub struct Qux<T>(T);

pub struct Bazzzz<T>(T);

pub trait Baz {}
impl<T> Baz for Bazzzz<T> {}

impl<T> Qux<T> {
    fn baz(self) -> Box<dyn Baz> {
        Box::new(Bazzzz(self.0)) //~ ERROR the parameter type `T` may not live long enough
    }
}

fn main() {
    let a = 10;
    let _b = bb(&a);
    let _c = cc(&a);
}
