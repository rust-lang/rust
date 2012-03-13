// aux-build:auto_serialize_lib.rs
// xfail-fast:aux-build currently incompatible

use std;
use auto_serialize_lib;
import std::prettyprint::serializer;
import std::ebml::serializer;
import std::ebml::deserializer;
import auto_serialize_lib::*;

#[auto_serialize]
enum expr {
    val(uint),
    plus(@expr, @expr),
    minus(@expr, @expr)
}

fn main() {
    test_ser_and_deser(plus(@minus(@val(3u), @val(10u)),
                            @plus(@val(22u), @val(5u))),
                       "plus(@minus(@val(3u), @val(10u)), \
                        @plus(@val(22u), @val(5u)))",
                       expr::serialize(_, _),
                       expr::deserialize(_),
                       expr::serialize(_, _));
}