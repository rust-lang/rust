//@ compile-flags: -Znext-solver=globally
//@ edition: 2021
#![crate_name = "foo"]

//@ has 'foo/struct.Foo.html'
//@ has - '//h3[@class="code-header"]' "impl<'a, 'b, T> Send for Foo<'a, 'b, T>where &'a T: Send, &'b T: Send"
pub struct Foo<'a, 'b, T: 'a + 'b>(&'a T, &'b T);
