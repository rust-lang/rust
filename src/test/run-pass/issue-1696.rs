use std;
import std::map;

fn main() {
    let m = map::new_bytes_hash();
    m.insert(str::bytes("foo"), str::bytes("bar"));
    log(error, m);
}
