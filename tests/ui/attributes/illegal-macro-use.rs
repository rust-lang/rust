// issue#140255

#[macro_use::a]       //~ ERROR cannot find
fn f0() {}

#[macro_use::a::b]    //~ ERROR cannot find
fn f1() {}

#[macro_escape::a]    //~ ERROR cannot find
fn f2() {}

#[macro_escape::a::b] //~ ERROR cannot find
fn f3() {}

fn main() {}
