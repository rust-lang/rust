// run-pass

fn main() {
    struct A;
    const FOO: &A = &(A as A);
    let _x = FOO;
}
