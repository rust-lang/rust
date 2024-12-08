//@ check-pass

// https://github.com/rust-lang/rust/issues/113257

#![deny(trivial_casts)] // The casts here are not trivial.

struct Foo<'a> { a: &'a () }

fn extend_lifetime_very_very_safely<'a>(v: *const Foo<'a>) -> *const Foo<'static> {
    // This should pass because raw pointer casts can do anything they want.
    v as *const Foo<'static>
}

trait Trait {}

fn assert_static<'a>(ptr: *mut (dyn Trait + 'a)) -> *mut (dyn Trait + 'static) {
    ptr as _
}

fn main() {
    let unit = ();
    let foo = Foo { a: &unit };
    let _long: *const Foo<'static> = extend_lifetime_very_very_safely(&foo);
}
