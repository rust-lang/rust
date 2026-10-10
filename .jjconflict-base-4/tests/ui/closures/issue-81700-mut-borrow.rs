//@ edition:2015..2021
fn foo(x: &mut u32) {
    let bar = || { foo(x); };
    bar(); //~ ERROR cannot borrow
}
fn main() {}
