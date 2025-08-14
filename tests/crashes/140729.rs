//@ known-bug: #140729
#![feature(min_generic_const_args)]

const C: usize = 0;
pub struct A<const M: usize> {}
impl A<C> {
    fn fun1() {}
}
impl A {
    fn fun1() {}
}
