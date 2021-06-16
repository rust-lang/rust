use std::ops::Deref;

pub struct A;
pub struct B;

// @has recursive_deref/struct.A.html '//code' 'impl Deref for A'
impl Deref for A {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

// @has recursive_deref/struct.B.html '//code' 'impl Deref for B'
impl Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}
