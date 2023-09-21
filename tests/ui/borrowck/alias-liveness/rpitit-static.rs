// check-pass

trait Foo {
    fn rpitit(&mut self) -> impl Sized + 'static;
}

fn test<T: Foo>(mut t: T) {
    let a = t.rpitit();
    let b = t.rpitit();
}

fn main() {}
