// xfail-stage0
// Tests that trans_path checks whether a
// pattern-bound var is an upvar (when translating
// the for-each body)
use std;
import std::option::*;
import std::uint;

fn foo(uint src) {

    alt (some(src)) {
        case (some(?src_id)) {
          for each (uint i in uint::range(0u, 10u)) {
            auto yyy = src_id;
            assert (yyy == 0u);
          }
        }
        case (_) {}
    }
}

fn main() {
  foo(0u);
}
