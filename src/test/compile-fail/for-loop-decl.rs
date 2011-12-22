// error-pattern: mismatched types
use std;
import std::map::hashmap;
import std::bitv;

type fn_info = {vars: hashmap<uint, var_info>};
type var_info = {a: uint, b: uint};

fn bitv_to_str(enclosing: fn_info, v: bitv::t) -> str {
    let s = "";

    // error is that the value type in the hash map is var_info, not a box
    enclosing.vars.values {|val|
        if bitv::get(v, val) { s += "foo"; }
    }
    ret s;
}

fn main() { #debug("OK"); }
