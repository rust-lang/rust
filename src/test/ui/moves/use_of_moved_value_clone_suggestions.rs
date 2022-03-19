// run-rustfix
#![allow(dead_code)]

// `Rc` is not ever `Copy`, we should not suggest adding `T: Copy` constraint.
// But should suggest adding `.clone()`.
fn move_rc<T>(t: std::rc::Rc<T>) {
    [t, t]; //~ use of moved value: `t`
}

// Even though `T` could be `Copy` it's already `Clone` so don't suggest adding `T: Copy` constraint,
// instead suggest adding `.clone()`.
fn move_clone_already<T: Clone>(t: T) {
    [t, t]; //~ use of moved value: `t`
}

// Same as `Rc`
fn move_clone_only<T>(t: (T, String)) {
    [t, t]; //~ use of moved value: `t`
}

// loop
fn move_in_a_loop<T: Clone>(t: T) {
    loop {
        if true {
            drop(t); //~ use of moved value: `t`
        } else {
            drop(t);
        }
    }
}

fn main() {}
