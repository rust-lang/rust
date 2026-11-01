// Regression test for https://github.com/rust-lang/rust/issues/67991
// The weird indentation is to exercise the span context that shows the enclosing items in the
// diagnostics being emitted.
trait Trait<
    T = (),
    K = Self,
> {
    fn foo()
    -> T;
    fn bar()
    -> K;
}

struct S;

impl Trait for S {
    fn foo() -> u32 { 0 } //~ ERROR
    fn bar() -> u32 { 0 } //~ ERROR
}
impl Trait<i32> for () {
    fn foo() -> u32 { 0 } //~ ERROR
    fn bar() -> u32 { 0 } //~ ERROR
}

fn main() {}
