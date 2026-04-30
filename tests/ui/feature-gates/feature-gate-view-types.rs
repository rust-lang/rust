struct Foo {
    a: usize,
    b: usize,
}

fn bar(a: &mut Foo.{ a }, b: &mut Foo.{ b }) {
    //~^ ERROR view types are experimental
    //~| ERROR view types are experimental
    a.a += 1;
    b.b += 1;
}

fn main() {
    let mut foo = Foo { a: 0, b: 0 };
    bar(&mut foo, &mut foo);
    //~^ ERROR cannot borrow `foo` as mutable more than once at a time
}
