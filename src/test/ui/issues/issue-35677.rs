use std::collections::HashMap;
fn intersect_map<K, V>(this: &mut HashMap<K, V>, other: HashMap<K, V>) -> bool {
    this.drain()
    //~^ ERROR no method named
}

fn main() {}
