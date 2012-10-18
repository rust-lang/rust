fn foo<T>() {
    struct foo {
        mut x: T, //~ ERROR attempt to use a type argument out of scope
        //~^ ERROR use of undeclared type name
        drop { }
    }
}
fn main() { }
