use std;
import std::prettyprint::serializer;
import std::io;

// Test where we link various types used by name.

#[auto_serialize]
type uint_vec = [uint];

#[auto_serialize]
type some_rec = {v: uint_vec};

#[auto_serialize]
enum an_enum = some_rec;

fn main() {
    let x = an_enum({v: [1u, 2u, 3u]});
    an_enum::serialize(io::stdout(), x);
    let s = io::with_str_writer {|w| an_enum::serialize(w, x)};
    #debug["s == %?", s];
    assert s == "an_enum({v: [1u, 2u, 3u]})";
}