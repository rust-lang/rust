//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

// Testing the errors in case we've made a wrong choice when
// calling an opaque.

use std::ops::Deref;

fn item_bound_is_too_weak() -> impl FnOnce() {
    if false {
        let mut var = item_bound_is_too_weak();
        var();
        var();
        //~^ ERROR use of moved value: `var`
    }

    let mut state = String::new();
    move || state.push('a')
}

fn opaque_type_no_impl_fn() -> impl Sized {
    if false {
        opaque_type_no_impl_fn()();
        //[current]~^ ERROR expected function, found `impl Sized`
        //[next]~^^ ERROR expected function, found `_`
    }

    1
}

fn opaque_type_no_impl_fn_incorrect() -> impl Sized {
    if false {
        opaque_type_no_impl_fn_incorrect()();
        //[current]~^ ERROR expected function, found `impl Sized`
        //[next]~^^ ERROR expected function, found `_`
    }

    || ()
}

fn opaque_type_deref_no_impl_fn() -> impl Deref<Target = impl Sized> {
    if false {
        opaque_type_deref_no_impl_fn()();
        //[current]~^ ERROR expected function, found `impl Deref<Target = impl Sized>`
        //[next]~^^ ERROR expected function, found `_`
    }

    &1
}

fn main() {}
