// xfail-stage0
// error-pattern:Attempt to use a type argument out of scope
fn foo[T] (&T x) {
    fn bar(fn (&T) -> T f) { };
}
fn main() { foo(1); }
