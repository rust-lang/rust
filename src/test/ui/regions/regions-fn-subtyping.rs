// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]
// Issue #2263.

// pretty-expanded FIXME #23616

#![allow(unused_variables)]

// Should pass region checking.
fn ok(f: Box<dyn FnMut(&usize)>) {
    // Here, g is a function that can accept a usize pointer with
    // lifetime r, and f is a function that can accept a usize pointer
    // with any lifetime.  The assignment g = f should be OK (i.e.,
    // f's type should be a subtype of g's type), because f can be
    // used in any context that expects g's type.  But this currently
    // fails.
    let mut g: Box<dyn for<'r> FnMut(&'r usize)> = Box::new(|x| { });
    g = f;
}

// This version is the same as above, except that here, g's type is
// inferred.
fn ok_inferred(f: Box<dyn FnMut(&usize)>) {
    let mut g: Box<dyn for<'r> FnMut(&'r usize)> = Box::new(|_| {});
    g = f;
}

pub fn main() {
}
