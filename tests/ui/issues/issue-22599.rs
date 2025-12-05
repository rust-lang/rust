#![deny(unused_variables)]

fn f(_: i32) {}

fn main() {
    let mut v = 0;
    f(v);
    v = match 0 { a => 0 }; //~ ERROR: unused variable: `a`
    f(v);
}
