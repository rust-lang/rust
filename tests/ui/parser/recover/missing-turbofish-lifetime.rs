//! Regression test for <https://github.com/rust-lang/rust/issues/162656>
//!
//! Test that omitting the turbofish when passing a lifetime to an expression
//! emits a targeted suggestion rather than a cascading syntax error.

#![allow(dead_code)]

//@ run-rustfix

struct Struct<'a> {
    string: &'a str,
}

fn struct_with_reserved_lifetime() {
    let _ = Struct<'_> {
        //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
        //~| HELP add `::`
        string: "",
    };
}

fn struct_with_named_lifetime<'a>() {
    let _ = Struct<'a> {
        //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
        //~| HELP add `::`
        string: "",
    };
}

fn struct_with_multichar_lifetime<'abc>() {
    let _ = Struct<'abc> {
        //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
        //~| HELP add `::`
        string: "",
    };
}

struct TupleStruct<'a>(&'a str);

fn tuple_struct_with_reserved_lifetime() {
    let _ = TupleStruct<'_>("");
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`
}

struct Wrapper<'a>(&'a str);
impl<'a> Wrapper<'a> { fn new(s: &'a str) -> Self { Self(s) } }

struct Struct2<'a> { x: i32, _p: std::marker::PhantomData<&'a ()> }
impl<'a> Struct2<'a> { fn method(&self) -> Option<()> { Some(()) } }

fn f<'a: 'a>() {}

struct StructT<'a, T> { a: &'a str, b: T }
struct Struct3<'a, 'b> { a: &'a str, b: &'b str }

fn chaining_cases<'a, 'b, T>() -> Option<()> {
    let _ = Wrapper<'a>::new("hi");
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    let _ = Struct2<'a> { x: 1, _p: std::marker::PhantomData }.method();
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    let _ = Struct2<'a> { x: 1, _p: std::marker::PhantomData }.method()?;
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    f<'_>();
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    // struct with two lifetimes
    let _ = Struct3<'a, 'b> { a: "hi", b: "hi" };
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    // struct with lifetime and type
    let _ = StructT<'a, &str> { a: "hi", b: "hi" };
    //~^ ERROR use `::<...>` instead of `<...>` to specify lifetime arguments
    //~| HELP add `::`

    Some(())
}

fn main() {}
