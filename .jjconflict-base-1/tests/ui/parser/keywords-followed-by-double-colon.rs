fn main() {
    struct::foo();
    //~^ ERROR expected identifier
}
fn bar() {
    mut::baz();
    //~^ ERROR expected expression, found keyword `mut`
}
