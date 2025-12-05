fn foo(s: &i32) -> &i32 {
    let xs;
    xs //~ ERROR: isn't initialized
}
fn main() {
    let y;
    if foo == y {}
    //~^ ERROR binary operation `==` cannot be applied to type
}
