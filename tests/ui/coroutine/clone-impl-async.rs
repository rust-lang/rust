//@ edition:2021
//@compile-flags: --diagnostic-width=300
// gate-test-coroutine_clone
// Verifies that feature(coroutine_clone) doesn't allow async blocks to be cloned/copied.

#![feature(coroutines, coroutine_clone)]

use std::future::ready;

struct NonClone;

fn local() {
    let inner_non_clone = async {
        let non_clone = NonClone;
        let () = ready(()).await;
        drop(non_clone);
    };
    check_copy(&inner_non_clone);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&inner_non_clone);
    //~^ ERROR : Clone` is not satisfied

    let non_clone = NonClone;
    let outer_non_clone = async move {
        drop(non_clone);
    };
    check_copy(&outer_non_clone);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&outer_non_clone);
    //~^ ERROR : Clone` is not satisfied

    let maybe_copy_clone = async move {};
    check_copy(&maybe_copy_clone);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&maybe_copy_clone);
    //~^ ERROR : Clone` is not satisfied
}

fn non_local() {
    let inner_non_clone_fn = the_inner_non_clone_fn();
    check_copy(&inner_non_clone_fn);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&inner_non_clone_fn);
    //~^ ERROR : Clone` is not satisfied

    let outer_non_clone_fn = the_outer_non_clone_fn(NonClone);
    check_copy(&outer_non_clone_fn);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&outer_non_clone_fn);
    //~^ ERROR : Clone` is not satisfied

    let maybe_copy_clone_fn = the_maybe_copy_clone_fn();
    check_copy(&maybe_copy_clone_fn);
    //~^ ERROR : Copy` is not satisfied
    check_clone(&maybe_copy_clone_fn);
    //~^ ERROR : Clone` is not satisfied
}

async fn the_inner_non_clone_fn() {
    let non_clone = NonClone;
    let () = ready(()).await;
    drop(non_clone);
}

async fn the_outer_non_clone_fn(non_clone: NonClone) {
    let () = ready(()).await;
    drop(non_clone);
}

async fn the_maybe_copy_clone_fn() {}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}

fn main() {}
