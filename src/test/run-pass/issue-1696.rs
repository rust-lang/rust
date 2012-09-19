extern mod std;
use std::map;
use std::map::HashMap;

fn main() {
    let m = map::HashMap();
    m.insert(str::to_bytes(~"foo"), str::to_bytes(~"bar"));
    log(error, m);
}
