// error-pattern: mismatched types
use std;
import std::map::hashmap;
import std::bitv;

type fn_info = {vars: hashmap<uint, var_info>};
type var_info = {a: uint, b: uint};

fn bitv_to_str(enclosing: fn_info, v: bitv::t) -> str {
    let s = "";

    // error is that the value type in the hash map is var_info, not a box
    for each p: @{key: uint, val: @uint} in enclosing.vars.items() {
        if bitv::get(v, *p.val) { s += "foo"; }
    }
    ret s;
}

fn main() { log "OK"; }
