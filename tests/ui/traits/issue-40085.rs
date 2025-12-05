//@ run-pass
use std::ops::Index;
fn bar() {}
static UNIT: () = ();
struct S;
impl Index<fn()> for S {
    type Output = ();
    fn index(&self, _: fn()) -> &() { &UNIT }
}
fn main() {
    S.index(bar);
    S[bar];
}
