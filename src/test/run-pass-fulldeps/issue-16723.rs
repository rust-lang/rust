// ignore-stage1
// aux-build:issue-16723.rs
#![feature(plugin)]
#![plugin(issue_16723)]

multiple_items!();

impl Struct1 {
    fn foo() {}
}
impl Struct2 {
    fn foo() {}
}

fn main() {
    Struct1::foo();
    Struct2::foo();
    println!("hallo");
}
