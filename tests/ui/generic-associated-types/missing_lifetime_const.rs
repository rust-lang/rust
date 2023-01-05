trait Foo {
    type Assoc<'a, const N: usize>;
}

fn foo<T: Foo>() {
    let _: <T as Foo>::Assoc<3>;
      //~^ ERROR  this associated type
}

fn main() {}
