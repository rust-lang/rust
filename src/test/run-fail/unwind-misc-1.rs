// error-pattern:fail

use std;
import std::map;
import std::map::hashmap;
import uint;

fn main() {
    let count = @mut 0u;
    pure fn hash(s: &~[@~str]) -> uint {
        if vec::len(*s) > 0u && *s[0] == ~"boom" { fail; }
        return 10u;
    }
    pure fn eq(s: &~[@~str], t: &~[@~str]) -> bool {
        return *s == *t;
    }

    let map = map::hashmap(hash, eq);
    let mut arr = ~[];
    for uint::range(0u, 10u) |i| {
        arr += ~[@~"key stuff"];
        map.insert(arr, arr + ~[@~"value stuff"]);
    }
    map.insert(~[@~"boom"], ~[]);
}
