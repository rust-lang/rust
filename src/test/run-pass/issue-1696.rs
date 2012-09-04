use std;
import std::map;
import std::map::hashmap;

fn main() {
    let m = map::bytes_hash();
    m.insert(str::to_bytes(~"foo"), str::to_bytes(~"bar"));
    log(error, m);
}
