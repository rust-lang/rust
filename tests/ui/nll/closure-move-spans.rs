// check that moves due to a closure capture give a special note
//@ run-rustfix
#![allow(unused_variables, unused_must_use, dead_code)]

fn move_after_move(x: String) {
    || x;
    let y = x; //~ ERROR
}

fn borrow_after_move(x: String) {
    || x;
    let y = &x; //~ ERROR
}

fn borrow_mut_after_move(mut x: String) {
    || x;
    let y = &mut x; //~ ERROR
}

fn fn_ref<F: Fn()>(f: F) -> F { f }
fn fn_mut<F: FnMut()>(f: F) -> F { f }

fn main() {}
