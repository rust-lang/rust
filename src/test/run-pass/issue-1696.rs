use std;
import std::map;
import std::map::hashmap;

fn main() {
    let m = map::bytes_hash();
    m.insert(str::bytes("foo"), str::bytes("bar"));
    log(error, m);
}
