// error-pattern:can't use type parameters from outer function
fn foo<T>(x: T) {
    fn bar(f: Box<FnMut(T) -> T>) { }
}
fn main() { foo(1); }
