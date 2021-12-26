// compile-flags: --crate-type lib
trait Foo<'x> {}

fn or<'a>(first: dyn Foo<'a>) -> dyn Foo<'a> {
    //~^ ERROR: the size for values of type `(dyn Foo<'a> + 'static)` cannot be known at compilation time [E0277]
    //~| ERROR: return type cannot have an unboxed trait object [E0746]
    return Box::new(0);
}
