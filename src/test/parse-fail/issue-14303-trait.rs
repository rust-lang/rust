// compile-flags: -Z parse-only

trait Foo<'a, T, 'b> {}
//~^ ERROR lifetime parameters must be declared prior to type parameters
