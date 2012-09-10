// error-pattern:fail

use std;
use std::map;
use std::map::HashMap;

fn main() {
    let count = @mut 0u;
    let map = map::HashMap();
    let mut arr = ~[];
    for uint::range(0u, 10u) |i| {
        arr += ~[@~"key stuff"];
        map.insert(arr, arr + ~[@~"value stuff"]);
        if arr.len() == 5 {
            fail;
        }
    }
}
