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
type uint_vec = [uint];

#[auto_serialize]
type some_rec = {v: uint_vec};

#[auto_serialize]
enum an_enum = some_rec;

fn main() {
    test_ser_and_deser(an_enum({v: [1u, 2u, 3u]}),
                       "an_enum({v: [1u, 2u, 3u]})",
                       an_enum::serialize(_, _),
                       an_enum::deserialize(_),
                       an_enum::serialize(_, _));
}