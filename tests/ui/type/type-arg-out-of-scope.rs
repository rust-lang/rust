//@ error-pattern:can't use generic parameters from outer item
fn foo<T>(x: T) {
    fn bar(f: Box<dyn FnMut(T) -> T>) { }
}
fn main() { foo(1); }
