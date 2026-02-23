//@ no-prefer-dynamic
//@ compile-flags: -O

pub trait Compare {
    fn eq(self);
}

pub fn wrap<A: Compare>(a: A) {
    Compare::eq(a);
}
