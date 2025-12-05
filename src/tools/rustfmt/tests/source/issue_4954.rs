trait Foo {
  type Arg<'a>;
}

struct Bar<T>(T) where for<'a> T: Foo<Arg<'a> = ()>;
