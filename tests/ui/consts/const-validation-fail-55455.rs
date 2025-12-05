// https://github.com/rust-lang/rust/issues/55454
//@ build-pass (FIXME(62277): could be check-pass?)

struct This<T>(T);

const C: This<Option<&i32>> = This(Some(&1));

fn main() {
}
