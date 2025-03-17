//@ run-rustfix
#![allow(dead_code, mismatched_lifetime_syntaxes)]
#![deny(no_mangle_generic_items)]

#[export_name = "foo"]
pub fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[export_name = "bar"]
pub extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[export_name = "baz"]
pub fn baz(x: &i32) -> &i32 { x }

#[export_name = "qux"]
pub fn qux<'a>(x: &'a i32) -> &i32 { x }

pub struct Foo;

impl Foo {
    #[export_name = "foo"]
    pub fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "bar"]
    pub extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    pub fn baz(x: &i32) -> &i32 { x }

    #[export_name = "qux"]
    pub fn qux<'a>(x: &'a i32) -> &i32 { x }
}

trait Trait1 {
    fn foo<T>();
    extern "C" fn bar<T>();
    fn baz(x: &i32) -> &i32;
    fn qux<'a>(x: &'a i32) -> &i32;
}

impl Trait1 for Foo {
    #[export_name = "foo"]
    fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "bar"]
    extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    fn baz(x: &i32) -> &i32 { x }

    #[export_name = "qux"]
    fn qux<'a>(x: &'a i32) -> &i32 { x }
}

trait Trait2<T> {
    fn foo();
    fn foo2<U>();
    extern "C" fn bar();
    fn baz(x: &i32) -> &i32;
    fn qux<'a>(x: &'a i32) -> &i32;
}

impl<T> Trait2<T> for Foo {
    #[export_name = "foo"]
    fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "foo2"]
    fn foo2<U>() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    fn baz(x: &i32) -> &i32 { x } //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "qux"]
    fn qux<'a>(x: &'a i32) -> &i32 { x } //~ ERROR functions generic over types or consts must be mangled
}

pub struct Bar<T>(#[allow(dead_code)] T);

impl<T> Bar<T> {
    #[export_name = "foo"]
    pub fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "bar"]
    pub extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    pub fn baz<U>() {} //~ ERROR functions generic over types or consts must be mangled
}

impl Bar<i32> {
    #[export_name = "qux"]
    pub fn qux() {}
}

trait Trait3 {
    fn foo();
    extern "C" fn bar();
    fn baz<U>();
}

impl<T> Trait3 for Bar<T> {
    #[export_name = "foo"]
    fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "bar"]
    extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[export_name = "baz"]
    fn baz<U>() {} //~ ERROR functions generic over types or consts must be mangled
}

pub struct Baz<'a>(#[allow(dead_code)] &'a i32);

impl<'a> Baz<'a> {
    #[export_name = "foo"]
    pub fn foo() {}

    #[export_name = "bar"]
    pub fn bar<'b>(x: &'b i32) -> &i32 { x }
}

trait Trait4 {
    fn foo();
    fn bar<'a>(x: &'a i32) -> &i32;
}

impl Trait4 for Bar<i32> {
    #[export_name = "foo"]
    fn foo() {}

    #[export_name = "bar"]
    fn bar<'b>(x: &'b i32) -> &i32 { x }
}

impl<'a> Trait4 for Baz<'a> {
    #[export_name = "foo"]
    fn foo() {}

    #[export_name = "bar"]
    fn bar<'b>(x: &'b i32) -> &i32 { x }
}

trait Trait5<T> {
    fn foo();
}

impl Trait5<i32> for Foo {
    #[export_name = "foo"]
    fn foo() {}
}

impl Trait5<i32> for Bar<i32> {
    #[export_name = "foo"]
    fn foo() {}
}

fn main() {}
