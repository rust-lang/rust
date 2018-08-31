// FIXME: Something in panic handling fails validation with full-MIR
// compile-flags: -Zmir-emit-validate=0
//error-pattern: the evaluated program panicked

fn main() {
    assert_eq!(5, 6);
}
