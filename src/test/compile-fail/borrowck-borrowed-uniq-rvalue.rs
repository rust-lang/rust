//buggy.rs
extern mod std;
use std::map::HashMap;
use std::map;

fn main() {
    let buggy_map :HashMap<uint, &uint> =
      HashMap::<uint, &uint>();
    buggy_map.insert(42, ~1); //~ ERROR illegal borrow
    
    // but it is ok if we use a temporary
    let tmp = ~2;
    buggy_map.insert(43, tmp);
}
