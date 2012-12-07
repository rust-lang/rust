extern mod std;
use std::map::HashMap;
use std::map;

fn main() {
    let buggy_map :HashMap<uint, &uint> = HashMap::<uint, &uint>();
    let x = ~1;
    buggy_map.insert(42, x);
}
