/// Test for https://github.com/rust-lang/rust-clippy/issues/11230

fn main() {
    const A: &[for<'a> fn(&'a ())] = &[];
    for v in A.iter() {}
}
