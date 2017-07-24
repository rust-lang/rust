// error-pattern: the evaluated program panicked

#[derive(Debug)]
struct A;

fn main() {
    // can't use assert_eq, b/c that will try to print the pointer addresses with full MIR enabled
    assert!(&A as *const A as *const () == &() as *const _)
}
