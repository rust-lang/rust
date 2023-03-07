// https://github.com/rust-lang/rust/issues/55454
// build-pass (FIXME(62277): could be check-pass?)

#[derive(PartialEq)]
struct This<T>(T);

fn main() {
    This(Some(&1)) == This(Some(&1));
}
