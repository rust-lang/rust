use std::collections::HashMap;

fn main() {
    let mut map = HashMap::<u32, u32>::new();
    map[&0] = 1; //~ ERROR cannot assign
}
