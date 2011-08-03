// error-pattern:Attempt to use a type argument out of scope
fn foo[T](x: &T) {
    fn bar(f: fn(&T) -> T ) { }
}
fn main() { foo(1); }