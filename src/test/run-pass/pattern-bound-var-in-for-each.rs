// Tests that trans_path checks whether a
// pattern-bound var is an upvar (when translating
// the for-each body)
use std;
import std::option::*;
import std::uint;

fn foo(src: uint) {


    alt some(src) {
      some(src_id) {
        for each i: uint  in uint::range(0u, 10u) {
            let yyy = src_id;
            assert (yyy == 0u);
        }
      }
      _ { }
    }
}

fn main() { foo(0u); }