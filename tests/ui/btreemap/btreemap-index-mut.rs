use std::collections::BTreeMap;

fn main() {
    let mut map = BTreeMap::<u32, u32>::new();
    map[&0] = 1; //~ ERROR cannot assign
}
