//@ compile-flags: -Z stack-protector=basic

pub fn main(){}

//~? ERROR incorrect value `basic` for unstable option `stack-protector`
