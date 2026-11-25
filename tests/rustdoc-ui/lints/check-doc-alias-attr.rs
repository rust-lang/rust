#![crate_type = "lib"]

#[doc(alias = "foo")] // ok!
#[doc(alias("bar", "baz"))] // ok!
pub struct Bar;

#[doc(alias)] //~ ERROR
#[doc(alias = 0)] //~ ERROR
#[doc(alias = "\"")] //~ ERROR
#[doc(alias = "\n")] //~ ERROR
#[doc(alias = "
")] //~^ ERROR
#[doc(alias = "\t")] //~ ERROR
#[doc(alias = " hello")] //~ ERROR
#[doc(alias = "hello ")] //~ ERROR
#[doc(alias = "")] //~ ERROR
pub struct Foo;

#[doc(alias(0))] //~ ERROR
#[doc(alias("\""))] //~ ERROR
#[doc(alias("\n"))] //~ ERROR
#[doc(alias("
"))] //~^ ERROR
#[doc(alias("\t"))] //~ ERROR
#[doc(alias(" hello"))] //~ ERROR
#[doc(alias("hello "))] //~ ERROR
#[doc(alias(""))] //~ ERROR
pub struct Foo2;
