//@ revisions: yes no
//@ compile-flags: -Znext-solver
//@[yes] check-pass

#![feature(const_trait_impl)]

const trait Foo {
    fn method(&self);
}

const impl<T: [const] Foo> Foo for (T,) {
    fn method(&self) {}
}

#[cfg(yes)]
const impl Foo for () {
    fn method(&self) {}
}

#[cfg(no)]
impl Foo for () {
    fn method(&self) {}
}

const fn bar<T: [const] Foo>(t: T) -> impl [const] Foo {
    (t,)
}

const _: () = {
    let opaque = bar(());
    //[no]~^ ERROR the trait bound `(): const Foo` is not satisfied
    opaque.method();
    //[no]~^ ERROR the trait bound `(): const Foo` is not satisfied
    std::mem::forget(opaque);
};

fn main() {}
