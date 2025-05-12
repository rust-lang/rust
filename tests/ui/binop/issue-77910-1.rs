fn foo(s: &i32) -> &i32 {
    let xs;
    xs //~ ERROR: isn't initialized
}
fn main() {
    let y;
    // we shouldn't ice with the bound var here.
    assert_eq!(foo, y);
    //~^ ERROR binary operation `==` cannot be applied to type
    //~| ERROR `for<'a> fn(&'a i32) -> &'a i32 {foo}` doesn't implement `Debug`
}
