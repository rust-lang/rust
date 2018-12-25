fn foo<T>() {
    fn bar(b: T) { } //~ ERROR can't use type parameters from outer
}
fn main() { }
