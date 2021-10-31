use std::ops::Deref;

pub struct A {}
impl A { pub fn foo_a(&self) {} }

pub struct B {}
impl B { pub fn foo_b(&self) {} }

pub struct C {}
impl C { pub fn foo_c(&self) {} }

// @has recursive_deref_sidebar/struct.A.html '//div[@class="sidebar-links"]' 'foo_b'
impl Deref for A {
    type Target = B;
    fn deref(&self) -> &B { todo!() }
}

// @has recursive_deref_sidebar/struct.A.html '//div[@class="sidebar-links"]' 'foo_c'
impl Deref for B {
    type Target = C;
    fn deref(&self) -> &C { todo!() }
}
