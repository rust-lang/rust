//@ check-pass
//@ compile-flags: -Znext-solver -Zassumptions-on-binders

#![crate_type = "lib"]

// Slight derivative of `type_relation_binders_inside_solver-1.rs` except this time the
// rigid alias is the opaque type `(impl Sized)<for<'b> fn(&'b ()) -> &'?0 ()>` instead.

fn foo<T>(_: T) -> impl Sized + use<T> {}

fn mk<'a>() -> for<'b> fn(&'b ()) -> &'a () { loop {} }

fn bar() {
    foo(mk());
}
