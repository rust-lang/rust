#[derive(PartialEq)]
struct Thing(usize);
impl PartialOrd for Thing { //~ ERROR not all trait items implemented, missing: `partial_cmp`
    fn le(&self, other: &Thing) -> bool { true }
    fn ge(&self, other: &Thing) -> bool { true }
}
fn main() {}
