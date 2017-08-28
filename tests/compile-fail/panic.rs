// FIXME: Probably failing due to https://github.com/solson/miri/issues/296
// compile-flags: -Zmir-emit-validate=0
//error-pattern: the evaluated program panicked

fn main() {
    assert_eq!(5, 6);
}
