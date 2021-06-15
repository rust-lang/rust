use std::ops::Deref;

pub struct A;
pub struct B;

// @has issue_85095/struct.A.html '//code' 'impl Deref for A'
impl Deref for A {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

// @has issue_85095/struct.B.html '//code' 'impl Deref for B'
impl Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}
