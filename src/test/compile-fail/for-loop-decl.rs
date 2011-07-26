// error-pattern: mismatched types
use std;
import std::map::hashmap;
import std::bitv;

type fn_info = rec(hashmap[uint, var_info] vars);
type var_info = rec(uint a, uint b);

fn bitv_to_str(fn_info enclosing, bitv::t v) -> str {
  auto s = "";

  // error is that the value type in the hash map is var_info, not a box
  for each (@rec(uint key, @uint val) p
            in enclosing.vars.items()) {
    if (bitv::get(v, *p.val)) {
      s += "foo";
    }
  }
  ret s;
}

fn main() {
  log "OK";
}