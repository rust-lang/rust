#![allow(dead_code)]

fn foo() {
    // Issue #52544
    let i: &i64 = &1;
    if i < 0 {}
    //~^ ERROR mismatched types [E0308]
}

fn bar() {
    // Issue #40660
    let foo = &&0;

    // Dereference LHS
    _ = foo == 0;
    //~^ERROR can't compare `&&{integer}` with `{integer}` [E0277]
    _ = foo == &0;
    //~^ERROR can't compare `&{integer}` with `{integer}` [E0277]
    _ = &&&&foo == 0;
    //~^ERROR can't compare `&&&&&&{integer}` with `{integer}` [E0277]
    _ = *foo == 0;
    //~^ERROR can't compare `&{integer}` with `{integer}` [E0277]
    _ = &&foo == &&0;
    //~^ERROR can't compare `&&{integer}` with `{integer}` [E0277]
    _ = &Box::new(42) == 42;
    //~^ERROR can't compare `&Box<{integer}>` with `{integer}` [E0277]
    _ = &Box::new(&Box::new(&42)) == 42;
    //~^ERROR can't compare `&Box<&Box<&{integer}>>` with `{integer}` [E0277]

    // Dereference RHS
    _ = 0 == foo;
    //~^ERROR can't compare `{integer}` with `&&{integer}` [E0277]
    _ = &0 == foo;
    //~^ERROR can't compare `{integer}` with `&{integer}` [E0277]
    _ = 0 == &&&&foo;
    //~^ERROR can't compare `{integer}` with `&&&&&&{integer}` [E0277]
    _ = 0 == *foo;
    //~^ERROR can't compare `{integer}` with `&{integer}` [E0277]
    _ = &&0 == &&foo;
    //~^ERROR can't compare `{integer}` with `&&{integer}` [E0277]

    // Dereference both sides
    _ = &Box::new(Box::new(42)) == &foo;
    //~^ERROR can't compare `Box<Box<{integer}>>` with `&&{integer}` [E0277]
    _ = &Box::new(42) == &foo;
    //~^ERROR can't compare `Box<{integer}>` with `&&{integer}` [E0277]
    _ = &Box::new(Box::new(Box::new(Box::new(42)))) == &foo;
    //~^ERROR can't compare `Box<Box<Box<Box<{integer}>>>>` with `&&{integer}` [E0277]
    _ = &foo == &Box::new(Box::new(Box::new(Box::new(42))));
    //~^ERROR can't compare `&&{integer}` with `Box<Box<Box<Box<{integer}>>>>` [E0277]

    // Don't suggest dereferencing the LHS; suggest boxing the RHS instead
    _ = Box::new(42) == 42;
    //~^ERROR mismatched types [E0308]

    // Don't suggest dereferencing with types that can't be compared
    struct Foo;
    _ = &&0 == Foo;
    //~^ERROR can't compare `&&{integer}` with `Foo` [E0277]
    _ = Foo == &&0;
    //~^ERROR binary operation `==` cannot be applied to type `Foo` [E0369]
}

fn baz() {
    // Issue #44695
    let owned = "foo".to_owned();
    let string_ref = &owned;
    let partial = "foobar";
    _ = string_ref == partial[..3];
    //~^ERROR can't compare `&String` with `str` [E0277]
    _ = partial[..3] == string_ref;
    //~^ERROR can't compare `str` with `&String` [E0277]
}

fn qux() {
    // Issue #119352
    const FOO: i32 = 42;
    let _ = FOO & (*"Sized".to_string().into_boxed_str());
    //~^ ERROR the size for values of type `str` cannot be known at compilation time
    //~| ERROR no implementation for `i32 & str` [E0277]
}

fn main() {}
