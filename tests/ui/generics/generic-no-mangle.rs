//@ run-rustfix
#![allow(dead_code, elided_named_lifetimes)]
#![deny(no_mangle_generic_items)]

#[no_mangle]
pub fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[no_mangle]
pub extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

#[no_mangle]
pub fn baz(x: &i32) -> &i32 { x }

#[no_mangle]
pub fn qux<'a>(x: &'a i32) -> &i32 { x }

pub struct Foo;

impl Foo {
    #[no_mangle]
    pub fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    pub extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    pub fn baz(x: &i32) -> &i32 { x }

    #[no_mangle]
    pub fn qux<'a>(x: &'a i32) -> &i32 { x }
}

trait Trait1 {
    fn foo<T>();
    extern "C" fn bar<T>();
    fn baz(x: &i32) -> &i32;
    fn qux<'a>(x: &'a i32) -> &i32;
}

impl Trait1 for Foo {
    #[no_mangle]
    fn foo<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    extern "C" fn bar<T>() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    fn baz(x: &i32) -> &i32 { x }

    #[no_mangle]
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
    #[no_mangle]
    fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    fn foo2<U>() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    fn baz(x: &i32) -> &i32 { x } //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    fn qux<'a>(x: &'a i32) -> &i32 { x } //~ ERROR functions generic over types or consts must be mangled
}

pub struct Bar<T>(#[allow(dead_code)] T);

impl<T> Bar<T> {
    #[no_mangle]
    pub fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    pub extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    pub fn baz<U>() {} //~ ERROR functions generic over types or consts must be mangled
}

impl Bar<i32> {
    #[no_mangle]
    pub fn qux() {}
}

trait Trait3 {
    fn foo();
    extern "C" fn bar();
    fn baz<U>();
}

impl<T> Trait3 for Bar<T> {
    #[no_mangle]
    fn foo() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    extern "C" fn bar() {} //~ ERROR functions generic over types or consts must be mangled

    #[no_mangle]
    fn baz<U>() {} //~ ERROR functions generic over types or consts must be mangled
}

pub struct Baz<'a>(#[allow(dead_code)] &'a i32);

impl<'a> Baz<'a> {
    #[no_mangle]
    pub fn foo() {}

    #[no_mangle]
    pub fn bar<'b>(x: &'b i32) -> &i32 { x }
}

trait Trait4 {
    fn foo();
    fn bar<'a>(x: &'a i32) -> &i32;
}

impl Trait4 for Bar<i32> {
    #[no_mangle]
    fn foo() {}

    #[no_mangle]
    fn bar<'b>(x: &'b i32) -> &i32 { x }
}

impl<'a> Trait4 for Baz<'a> {
    #[no_mangle]
    fn foo() {}

    #[no_mangle]
    fn bar<'b>(x: &'b i32) -> &i32 { x }
}

trait Trait5<T> {
    fn foo();
}

impl Trait5<i32> for Foo {
    #[no_mangle]
    fn foo() {}
}

impl Trait5<i32> for Bar<i32> {
    #[no_mangle]
    fn foo() {}
}

fn main() {}
