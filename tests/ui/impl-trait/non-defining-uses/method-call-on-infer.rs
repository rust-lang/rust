//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

fn fn_trait() -> impl Fn() {
    if false {
        let f = fn_trait();
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

fn fn_once() -> impl FnOnce() {
    if false {
        let mut f = fn_once();
        f();
    }
    
    let string = String::new();
    move || drop(string)
}

fn main() {}
