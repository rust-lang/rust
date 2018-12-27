// Test that the types of distinct fn items are not compatible by
// default. See also `run-pass/fn-item-type-*.rs`.

fn foo<T>(x: isize) -> isize { x * 2 }
fn bar<T>(x: isize) -> isize { x * 4 }

fn eq<T>(x: T, y: T) { }

trait Foo { fn foo() { /* this is a default fn */ } }
impl<T> Foo for T { /* `foo` is still default here */ }

fn main() {
    eq(foo::<u8>, bar::<u8>);
    //~^ ERROR mismatched types
    //~|  expected type `fn(isize) -> isize {foo::<u8>}`
    //~|  found type `fn(isize) -> isize {bar::<u8>}`
    //~|  expected fn item, found a different fn item

    eq(foo::<u8>, foo::<i8>);
    //~^ ERROR mismatched types
    //~| expected u8, found i8

    eq(bar::<String>, bar::<Vec<u8>>);
    //~^ ERROR mismatched types
    //~|  expected type `fn(isize) -> isize {bar::<std::string::String>}`
    //~|  found type `fn(isize) -> isize {bar::<std::vec::Vec<u8>>}`
    //~|  expected struct `std::string::String`, found struct `std::vec::Vec`

    // Make sure we distinguish between trait methods correctly.
    eq(<u8 as Foo>::foo, <u16 as Foo>::foo);
    //~^ ERROR mismatched types
    //~| expected u8, found u16
}
