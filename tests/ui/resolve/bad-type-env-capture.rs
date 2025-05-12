fn foo<T>() {
    fn bar(b: T) { } //~ ERROR can't use generic parameters from outer
}
fn main() { }
