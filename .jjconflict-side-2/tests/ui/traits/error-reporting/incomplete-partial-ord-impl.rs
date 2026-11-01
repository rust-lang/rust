//! Regression test for <https://github.com/rust-lang/rust/issues/3344>.
//! Unimplemented items in impl caused ICE.

#[derive(PartialEq)]
struct Thing(usize);
impl PartialOrd for Thing { //~ ERROR not all trait items implemented, missing: `partial_cmp`
    fn le(&self, other: &Thing) -> bool { true }
    fn ge(&self, other: &Thing) -> bool { true }
}
fn main() {}
