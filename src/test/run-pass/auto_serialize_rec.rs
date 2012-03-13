// aux-build:auto_serialize_lib.rs
// xfail-fast:aux-build currently incompatible

use std;
use auto_serialize_lib;
import std::prettyprint::serializer;
import std::ebml::serializer;
import std::ebml::deserializer;
import auto_serialize_lib::*;

#[auto_serialize]
type point = {x: uint, y: uint};

fn main() {
    test_ser_and_deser({x: 3u, y: 5u},
                       "{x: 3u, y: 5u}",
                       point::serialize(_, _),
                       point::deserialize(_),
                       point::serialize(_, _));
}