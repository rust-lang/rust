// run-pass
trait Trait { fn dummy(&self) { } }

#[derive(Debug)]
struct Foo<T: Trait> {
    foo: T,
}

#[derive(Debug)]
struct Bar<T> where T: Trait {
    bar: T,
}

impl Trait for isize {}

fn main() {
    let a = Foo { foo: 12 };
    let b = Bar { bar: 12 };
    println!("{:?} {:?}", a, b);
}
