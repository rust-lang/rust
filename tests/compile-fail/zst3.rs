// error-pattern: the evaluated program panicked

#[derive(Debug)]
struct A;

fn main() {
    assert_eq!(&A as *const A, &A as *const A);
}
