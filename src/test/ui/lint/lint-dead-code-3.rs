#![allow(unused_variables)]
#![allow(non_camel_case_types)]
#![deny(dead_code)]

#![crate_type="lib"]


pub use extern_foo as x;
extern {
    pub fn extern_foo();
}

struct Foo; //~ ERROR: struct is never constructed
impl Foo {
    fn foo(&self) { //~ ERROR: method is never used
        bar()
    }
}

fn bar() { //~ ERROR: function is never used
    fn baz() {}

    Foo.foo();
    baz();
}

// no warning
struct Foo2;
impl Foo2 { fn foo2(&self) { bar2() } }
fn bar2() {
    fn baz2() {}

    Foo2.foo2();
    baz2();
}

pub fn pub_fn() {
    let foo2_struct = Foo2;
    foo2_struct.foo2();

    blah::baz();
}

mod blah {
    // not warned because it's used in the parameter of `free` and return of
    // `malloc` below, which are also used.
    enum c_void {}

    extern {
        fn free(p: *const c_void);
        fn malloc(size: usize) -> *const c_void;
    }

    pub fn baz() {
        unsafe { free(malloc(4)); }
    }
}

enum c_void {} //~ ERROR: enum is never used
extern {
    fn free(p: *const c_void); //~ ERROR: foreign function is never used
}

// Check provided method
mod inner {
    pub trait Trait {
        fn f(&self) { f(); }
    }

    impl Trait for isize {}

    fn f() {}
}

pub fn foo() {
    let a: &dyn inner::Trait = &1_isize;
    a.f();
}
