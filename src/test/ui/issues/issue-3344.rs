#[derive(PartialEq)]
struct thing(usize);
impl PartialOrd for thing { //~ ERROR not all trait items implemented, missing: `partial_cmp`
    fn le(&self, other: &thing) -> bool { true }
    fn ge(&self, other: &thing) -> bool { true }
}
fn main() {}
