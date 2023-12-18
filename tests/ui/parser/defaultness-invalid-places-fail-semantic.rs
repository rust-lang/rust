#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

fn main() {}

trait X {
    default const A: u8; //~ ERROR `default` is only allowed on items in trait impls
    default const B: u8 = 0;  //~ ERROR `default` is only allowed on items in trait impls
    default type D; //~ ERROR `default` is only allowed on items in trait impls
    default type C: Ord; //~ ERROR `default` is only allowed on items in trait impls
    default fn f1(); //~ ERROR `default` is only allowed on items in trait impls
    default fn f2() {} //~ ERROR `default` is only allowed on items in trait impls
}

default const E: u8 = 0; //~ ERROR `default` is only allowed on items in trait impls
default type F = (); //~ ERROR `default` is only allowed on items in trait impls
default fn h() {} //~ ERROR `default` is only allowed on items in trait impls
