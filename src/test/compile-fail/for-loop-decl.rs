// error-pattern: mismatched types
use std;
import std::map::hashmap;
import std::bitv;

type fn_info = {vars: hashmap<uint, var_info>};
type var_info = {a: uint, b: uint};

fn bitv_to_str(enclosing: fn_info, v: ~bitv::bitv) -> str {
    let s = "";

    // error is that the value type in the hash map is var_info, not a box
    for enclosing.vars.each_value |val| {
        if v.get(val) { s += "foo"; }
    }
    return s;
}

fn main() { debug!{"OK"}; }
