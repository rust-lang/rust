// gate-test-const_generics

fn foo<const X: ()>() {} //~ ERROR const generics are unstable

fn main() {}
