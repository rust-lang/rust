// error-pattern:fail

use std;
import std::map;
import std::uint;

fn main() {
    let count = @mutable 0u;
    fn hash(&&s: [@str]) -> uint {
        if (std::vec::len(s) > 0u && std::str::eq(*s[0], "boom")) { fail; }
        ret 10u;
    }
    fn eq(&&s: [@str], &&t: [@str]) -> bool {
        ret s == t;
    }

    let map = map::mk_hashmap(hash, eq);
    let arr = [];
    uint::range(0u, 10u) {|i|
        arr += [@"key stuff"];
        map.insert(arr, arr + [@"value stuff"]);
    }
    map.insert([@"boom"], []);
}