use std::ops::Deref;

pub struct A;
pub struct B;

// @has recursive_deref/struct.A.html '//h3[@class="code-header in-band"]' 'impl Deref for A'
impl Deref for A {
    type Target = B;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}

// @has recursive_deref/struct.B.html '//h3[@class="code-header in-band"]' 'impl Deref for B'
impl Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        panic!()
    }
}
