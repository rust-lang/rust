fn foo() {}
fn main() {
    let mut y = 42;
    let x = &mut y;
    foo()
    *x = 0;  //~ ERROR invalid left-hand side of assignment
    //~^ ERROR cannot multiply `()` by `&mut {integer}`
    println!("{y}");
}
