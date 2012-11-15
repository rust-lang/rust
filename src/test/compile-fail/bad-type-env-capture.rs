fn foo<T>() {
    fn bar(b: T) { } //~ ERROR attempt to use a type argument out of scope
    //~^ ERROR use of undeclared type name
}
fn main() { }
