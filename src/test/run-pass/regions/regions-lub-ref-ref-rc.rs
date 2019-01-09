// run-pass
#![allow(dead_code)]
// Test a corner case of LUB coercion. In this case, one arm of the
// match requires a deref coercion and the other doesn't, and there
// is an extra `&` on the `rc`. We want to be sure that the lifetime
// assigned to this `&rc` value is not `'a` but something smaller.  In
// other words, the type from `rc` is `&'a Rc<String>` and the type
// from `&rc` should be `&'x &'a Rc<String>`, where `'x` is something
// small.

use std::rc::Rc;

#[derive(Clone)]
enum CachedMir<'mir> {
    Ref(&'mir String),
    Owned(Rc<String>),
}

impl<'mir> CachedMir<'mir> {
    fn get_ref<'a>(&'a self) -> &'a String {
        match *self {
            CachedMir::Ref(r) => r,
            CachedMir::Owned(ref rc) => &rc,
        }
    }
}

fn main() { }
