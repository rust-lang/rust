use std;
import std::prettyprint::serializer;
import std::io;

#[auto_serialize]
type point = {x: uint, y: uint};

fn main() {
    let s = io::with_str_writer {|w| point::serialize(w, {x: 3u, y: 5u}) };
    #debug["s == %?", s];
    assert s == "{x: 3u, y: 5u}";
}