//@ aux-build:no_method_suggested_traits.rs
//@ dont-require-annotations: HELP

extern crate no_method_suggested_traits;

struct Foo;
enum Bar { X }

mod foo {
    pub trait Bar {
        fn method(&self) {}

        fn method2(&self) {}
    }

    impl Bar for u32 {}

    impl Bar for char {}
}

fn main() {
    // test the values themselves, and autoderef.


    1u32.method();
    //~^ ERROR no method named
    //~| HELP items from traits can only be used if the trait is in scope
    std::rc::Rc::new(&mut Box::new(&1u32)).method();
    //~^ HELP items from traits can only be used if the trait is in scope
    //~| ERROR no method named `method` found for struct

    'a'.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&'a')).method();
    //~^ ERROR no method named

    1i32.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1i32)).method();
    //~^ ERROR no method named

    Foo.method();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Foo)).method();
    //~^ ERROR no method named

    1u64.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1u64)).method2();
    //~^ ERROR no method named

    no_method_suggested_traits::Foo.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Foo)).method2();
    //~^ ERROR no method named
    no_method_suggested_traits::Bar::X.method2();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Bar::X)).method2();
    //~^ ERROR no method named

    Foo.method3();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Foo)).method3();
    //~^ ERROR no method named
    Bar::X.method3();
    //~^ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&Bar::X)).method3();
    //~^ ERROR no method named

    // should have no help:
    1_usize.method3(); //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&1_usize)).method3(); //~ ERROR no method named
    no_method_suggested_traits::Foo.method3();  //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Foo)).method3();
    //~^ ERROR no method named
    no_method_suggested_traits::Bar::X.method3();  //~ ERROR no method named
    std::rc::Rc::new(&mut Box::new(&no_method_suggested_traits::Bar::X)).method3();
    //~^ ERROR no method named
}
