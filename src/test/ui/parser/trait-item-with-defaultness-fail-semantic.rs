#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

fn main() {}

trait X {
    default const A: u8; //~ ERROR `default` is only allowed on items in `impl` definitions
    default const B: u8 = 0;  //~ ERROR `default` is only allowed on items in `impl` definitions
    default type D; //~ ERROR `default` is only allowed on items in `impl` definitions
    default type C: Ord; //~ ERROR `default` is only allowed on items in `impl` definitions
    default fn f1(); //~ ERROR `default` is only allowed on items in `impl` definitions
    default fn f2() {} //~ ERROR `default` is only allowed on items in `impl` definitions
}
