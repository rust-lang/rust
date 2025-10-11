//@ revisions: current_ok next_ok current_ambig next_ambig
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next_ok] compile-flags: -Znext-solver
//@[next_ambig] compile-flags: -Znext-solver
//@[current_ok] check-pass
//@[next_ok] check-pass

// Regression test for trait-system-refactor-initiative#191.
trait Foo<'a> {
    type Input;
}

impl<'a, F: Fn(&'a u32)> Foo<'a> for F {
    type Input = &'a u32;
}

fn needs_super<F: for<'a> Fn(<F as Foo<'a>>::Input) + for<'a> Foo<'a>>(_: F) {}

fn main() {
    #[cfg(any(current_ok, next_ok))]
    needs_super(|_: &u32| {});
    #[cfg(any(current_ambig, next_ambig))]
    needs_super(|_| {});
    //[next_ambig]~^ ERROR expected a `Fn(&'a u32)` closure, found
    //[next_ambig]~| ERROR the trait bound
    //[current_ambig]~^^^ ERROR implementation of `Foo` is not general enough
    //[current_ambig]~| ERROR implementation of `Fn` is not general enough
    //[current_ambig]~| ERROR implementation of `Foo` is not general enough
    //[current_ambig]~| ERROR implementation of `FnOnce` is not general enough
    //[current_ambig]~| ERROR implementation of `Foo` is not general enough
}
