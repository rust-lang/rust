// ignore-tidy-linelength

#![crate_name = "foo"]
#![crate_type = "lib"]

pub struct Foo {
    pub some_func: for<'a> fn(val: &'a i32) -> i32,
    pub some_trait: dyn for<'a> Trait<'a>,
}

// @has foo/struct.Foo.html '//span[@id="structfield.some_func"]' "some_func: for<'a> fn(val: &'a i32) -> i32"
// @has foo/struct.Foo.html '//span[@id="structfield.some_trait"]' "some_trait: dyn Trait<'a>"

pub trait Trait<'a> {}
