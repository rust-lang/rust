//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

trait FnOnceForGenericRef<T>: FnOnce(&T) -> Self::FnOutput {
    type FnOutput;
}

impl<T, R, F: FnOnce(&T) -> R> FnOnceForGenericRef<T> for F {
    type FnOutput = R;
}

struct Data<T, D: FnOnceForGenericRef<T>> {
    value: Option<T>,
    output: Option<D::FnOutput>,
}

impl<T, D: FnOnceForGenericRef<T>> Data<T, D> {
    fn new(value: T, f: D) -> Self {
        let output = f(&value);
        Self { value: Some(value), output: Some(output) }
    }
}

fn test() {
    Data::new(String::new(), |_| {});
}

fn main() {}
