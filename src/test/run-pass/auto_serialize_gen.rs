// aux-build:auto_serialize_lib.rs
// xfail-fast:aux-build currently incompatible

use std;
use auto_serialize_lib;
import std::prettyprint::serializer;
import std::ebml::serializer;
import std::ebml::deserializer;
import auto_serialize_lib::*;

// Test where we link various types used by name.

#[auto_serialize]
type spanned<T> = {lo: uint, hi: uint, node: T};

#[auto_serialize]
type spanned_uint = spanned<uint>;

fn main() {
    test_ser_and_deser({lo: 0u, hi: 5u, node: 22u},
                       "{lo: 0u, hi: 5u, node: 22u}",
                       spanned_uint::serialize(_, _),
                       spanned_uint::deserialize(_),
                       spanned_uint::serialize(_, _));
}