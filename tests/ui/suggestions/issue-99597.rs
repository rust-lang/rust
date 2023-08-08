#![allow(dead_code)]

trait T1 { }

trait T2 {
    fn test(&self) { }
}

fn go(s: &impl T1) {
    //~^ SUGGESTION (
    s.test();
    //~^ ERROR no method named `test`
}

fn main() { }
