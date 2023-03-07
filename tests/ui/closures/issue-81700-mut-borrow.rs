fn foo(x: &mut u32) {
    let bar = || { foo(x); };
    bar(); //~ ERROR cannot borrow
}
fn main() {}
