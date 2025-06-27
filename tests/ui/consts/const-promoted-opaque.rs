//@revisions: string unit atomic
#![feature(type_alias_impl_trait)]

//! Check that we do not cause cycle errors when trying to
//! obtain information about interior mutability of an opaque type.
//! This used to happen, because when the body-analysis failed, we
//! checked the type instead, but the constant was also defining the
//! hidden type of the opaque type. Thus we ended up relying on the
//! result of our analysis to compute the result of our analysis.

pub type Foo = impl Sized;

#[cfg(string)]
#[define_opaque(Foo)]
const fn foo() -> Foo {
    String::new()
}

#[cfg(atomic)]
#[define_opaque(Foo)]
const fn foo() -> Foo {
    std::sync::atomic::AtomicU8::new(42)
}

#[cfg(unit)]
#[define_opaque(Foo)]
const fn foo() -> Foo {}

const FOO: Foo = foo();

const BAR: () = {
    let _: &'static _ = &FOO;
    //[string,atomic,unit]~^ ERROR: destructor of `Foo` cannot be evaluated at compile-time
};

const BAZ: &Foo = &FOO;
//[atomic]~^ ERROR: interior mutable shared borrows of lifetime-extended temporaries

fn main() {
    let _: &'static _ = &FOO;
    //[string,atomic,unit]~^ ERROR: temporary value dropped while borrowed
}
