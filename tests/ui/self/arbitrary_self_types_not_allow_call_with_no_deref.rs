#![feature(arbitrary_self_types)]

use std::rc::Rc;

struct Foo;

struct CppRef<T>(T);

impl<T> std::ops::Receiver for CppRef<T> {
    type Target = T;
}

impl Foo{
    fn frobnicate_self(self) {}
    fn frobnicate_ref(&self) {}
    fn frobnicate_cpp_ref(self: CppRef<Self>) {}
}

fn main() {
    let foo_rc = Rc::new(Foo);

    // this compiles fine, and desugars to `Foo::frobnicate_ref(&*foo_rc)`
    foo_rc.frobnicate_ref();

    let foo_cpp_ref = CppRef(Foo);

    // should not compile because it would desugar to `Foo::frobnicate_ref(&*foo_cpp_ref)`
    // and you can't deref a CppRef
    foo_cpp_ref.frobnicate_ref();
    //~^ ERROR no method named

    foo_cpp_ref.frobnicate_self(); // would desugar to `Foo::frobnicate_self(*foo_cpp_ref)`
    //~^ ERROR no method named

    // should compile, because we're not dereffing the CppRef
    // desugars to `Foo::frobnicate_cpp_ref(foo_cpp_ref)`
    foo_cpp_ref.frobnicate_cpp_ref();
}
