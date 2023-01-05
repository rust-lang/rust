// run-pass
struct A<F: FnOnce()->T,T>(F::Output);
struct B<F: FnOnce()->T,T>(A<F,T>);

// Removing Option causes it to compile.
fn foo<T,F: FnOnce()->T>(f: F) -> Option<B<F,T>> {
    Some(B(A(f())))
}

fn main() {
    let v = (|| foo(||4))();
    match v {
        Some(B(A(4))) => {},
        _ => unreachable!()
    }
}
