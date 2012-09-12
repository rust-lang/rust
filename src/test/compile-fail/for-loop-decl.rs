// error-pattern: mismatched types
extern mod std;
use std::map::HashMap;
use std::bitv;

type fn_info = {vars: HashMap<uint, var_info>};
type var_info = {a: uint, b: uint};

fn bitv_to_str(enclosing: fn_info, v: ~bitv::Bitv) -> str {
    let s = "";

    // error is that the value type in the hash map is var_info, not a box
    for enclosing.vars.each_value |val| {
        if v.get(val) { s += "foo"; }
    }
    return s;
}

fn main() { debug!("OK"); }
