// run-pass

// https://github.com/rust-lang/rust/issues/37448

fn main() {
    struct A;
    const FOO: &A = &(A as A);
    let _x = FOO;
}
