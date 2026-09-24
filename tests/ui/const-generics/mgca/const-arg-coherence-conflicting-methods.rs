// Regression test for #140729

#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

use std::gca;

const C: usize = gca!(0);
pub struct A<const M: usize> {}
impl A<C> {
    fn fun1() {}
    //~^ ERROR duplicate definitions with name `fun1`
}
impl A {
    //~^ ERROR missing generics for struct `A`
    fn fun1() {}
}

fn main() {}
