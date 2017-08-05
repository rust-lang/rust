// FIXME: Due to https://github.com/rust-lang/rust/issues/43457 we have to disable validation
// compile-flags: -Zmir-emit-validate=0

//error-pattern: the evaluated program panicked

fn main() {
    assert_eq!(5, 6);
}
