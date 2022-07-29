fn foo(s: &i32) -> &i32 {
    let xs;
    xs
}
fn main() {
    let y;
    // we shouldn't ice with the bound var here.
    assert_eq!(foo, y);
    //~^ ERROR binary operation `==` cannot be applied to type
    //~| ERROR `[fn item {foo}: for<'r> fn(&'r i32) -> &'r i32]` doesn't implement `Debug`
}
