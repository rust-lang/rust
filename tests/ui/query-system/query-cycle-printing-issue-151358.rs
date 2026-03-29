//@ compile-flags: -Zverbose-internals
//@ dont-require-annotations: ERROR
//@ normalize-stderr: "(compiler/[^ )]+\.rs):\d+:\d+" -> "$1:LL:CC"
trait Default {}
use std::num::NonZero;
fn main() {
    NonZero();
    format!("{}", 0);
}
