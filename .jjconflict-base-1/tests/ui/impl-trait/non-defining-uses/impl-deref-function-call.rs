//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#181. We want to
// be able to step through `impl Deref` in its defining scope.
use std::ops::{Deref, DerefMut};
fn impl_deref_fn() -> impl Deref<Target = fn(fn(&str) -> usize)> {
    if false {
        let func = impl_deref_fn();
        func(|s| s.len());
    }

    &((|_| ()) as fn(_))
}

fn impl_deref_impl_fn() -> impl Deref<Target = impl Fn()> {
    if false {
        let func = impl_deref_impl_fn();
        func();
    }

    &|| ()
}

fn impl_deref_impl_deref_impl_fn() -> impl Deref<Target = impl Deref<Target = impl Fn()>> {
    if false {
        let func = impl_deref_impl_deref_impl_fn();
        func();
    }

    &&|| ()
}


fn impl_deref_mut_impl_fn() -> impl DerefMut<Target = impl Fn()> {
    if false {
        let func = impl_deref_impl_fn();
        func();
    }

    Box::new(|| ())
}


fn impl_deref_mut_impl_fn_mut() -> impl DerefMut<Target = impl FnMut()> {
    if false {
        let mut func = impl_deref_mut_impl_fn_mut();
        func();
    }

    let mut state = 0;
    Box::new(move || state += 1)
}
fn main() {}
