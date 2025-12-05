//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for trait-system-refactor-initiative#181. Make sure calling
// opaque types works.

fn fn_trait() -> impl Fn() {
    if false {
        let f = fn_trait();
        f();
    }

    || ()
}

fn fn_trait_ref() -> impl Fn() {
    if false {
        let f = &fn_trait();
        f();
    }
    || ()
}

fn fn_mut() -> impl FnMut() -> usize {
    if false {
        let mut f = fn_mut();
        f();
    }

    let mut state = 0;
    move || {
        state += 1;
        state
    }
}

fn fn_mut_ref() -> impl FnMut() -> usize {
    if false {
        let mut f = &mut fn_mut();
        f();
    }

    let mut state = 0;
    move || {
        state += 1;
        state
    }
}


fn fn_once() -> impl FnOnce() {
    if false {
        let mut f = fn_once();
        f();
    }

    let string = String::new();
    move || drop(string)
}

fn fn_once_ref() -> impl FnOnce() {
    if false {
        let mut f = Box::new(fn_once_ref());
        f();
    }

    let string = String::new();
    move || drop(string)
}

fn main() {}
