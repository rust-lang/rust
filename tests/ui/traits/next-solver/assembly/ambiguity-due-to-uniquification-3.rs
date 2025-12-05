//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass

// Regression test from trait-system-refactor-initiative#27.
//
// Unlike in the previous two tests, `dyn Object<?x, ?y>: Trait<?x>` relies
// on structural identity of type inference variables. This inference variable
// gets constrained to a type containing a region later on. To prevent this
// from causing an ICE during MIR borrowck, we stash goals which depend on
// inference variables and then reprove them at the end of HIR typeck.

#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
trait Trait<T> {}
impl<T> Trait<T> for () {}

trait Object<T, U>: Trait<T> + Trait<U> {}

#[derive(Clone, Copy)]
struct Inv<T>(*mut T);
fn foo<T: Sized, U: Sized>() -> (Inv<dyn Object<T, U>>, Inv<T>) { todo!() }
fn impls_trait<T: Trait<U>, U>(_: Inv<T>, _: Inv<U>) {}

fn bar() {
    let (obj, t) = foo();
    impls_trait(obj, t);
    //[next]~^ ERROR type annotations needed
    let _: Inv<dyn Object<&(), &()>> = obj;
}

fn main() {}
