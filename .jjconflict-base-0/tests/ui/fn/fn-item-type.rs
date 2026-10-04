// Test that the types of distinct fn items are not compatible by
// default. See also `run-pass/fn-item-type-*.rs`.

//@ dont-require-annotations: NOTE

fn foo<T>(x: isize) -> isize {
    x * 2
}
fn bar<T>(x: isize) -> isize {
    x * 4
}

fn eq<T>(x: T, y: T) {}

trait Foo {
    fn foo() { /* this is a default fn */
    }
}
impl<T> Foo for T {
    /* `foo` is still default here */
}

fn main() {
    eq(foo::<u8>, bar::<u8>);
    //~^ ERROR mismatched types
    //~| NOTE expected fn item `fn(_) -> _ {foo::<u8>}`
    //~| NOTE found fn item `fn(_) -> _ {bar::<u8>}`
    //~| NOTE expected fn item, found a different fn item
    //~| NOTE different fn items have unique types, even if their signatures are the same

    eq(foo::<u8>, foo::<i8>);
    //~^ ERROR mismatched types
    //~| NOTE expected `u8`, found `i8`
    //~| NOTE different fn items have unique types, even if their signatures are the same

    eq(bar::<String>, bar::<Vec<u8>>);
    //~^ ERROR mismatched types
    //~| NOTE found fn item `fn(_) -> _ {bar::<Vec<u8>>}`
    //~| NOTE expected `String`, found `Vec<u8>`

    // Make sure we distinguish between trait methods correctly.
    eq(<u8 as Foo>::foo, <u16 as Foo>::foo);
    //~^ ERROR mismatched types
    //~| NOTE expected `u8`, found `u16`
    //~| NOTE different fn items have unique types, even if their signatures are the same

    eq(foo::<u8>, bar::<u8> as fn(isize) -> isize);
    //~^ ERROR mismatched types
    //~| NOTE found fn pointer `fn(_) -> _`
    //~| NOTE expected fn item, found fn pointer

    eq(foo::<u8> as fn(isize) -> isize, bar::<u8>); // ok!
}
