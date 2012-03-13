// aux-build:auto_serialize_lib.rs
// xfail-fast:aux-build currently incompatible

use std;
use auto_serialize_lib;
import std::prettyprint::serializer;
import std::ebml::serializer;
import std::ebml::deserializer;
import auto_serialize_lib::*;

#[auto_serialize]
type uint_vec = [uint];

fn main() {
    test_ser_and_deser([1u, 2u, 3u],
                       "[1u, 2u, 3u]",
                       uint_vec::serialize(_, _),
                       uint_vec::deserialize(_),
                       uint_vec::serialize(_, _));
}