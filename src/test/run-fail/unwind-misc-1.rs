// error-pattern:fail

use std;
import std::map;
import std::uint;

fn main() {
    let count = @mutable 0u;
    let hash = bind fn (&&_s: [@str], count: @mutable uint) -> uint {
        *count += 1u;
        if *count == 10u {
            fail;
        } else {
            ret *count;
        }
    } (_, count);

    fn eq(&&s: [@str], &&t: [@str]) -> bool {
        ret s == t;
    }

    let map = map::mk_hashmap(hash, eq);
    let arr = [];
    uint::range(0u, 10u) {|i|
        arr += [@"key stuff"];
        map.insert(arr, arr + [@"value stuff"]);
    }
}