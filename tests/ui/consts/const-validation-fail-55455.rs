// https://github.com/rust-lang/rust/issues/55454
//@ check-pass

struct This<T>(T);

const C: This<Option<&i32>> = This(Some(&1));

fn main() {
}
