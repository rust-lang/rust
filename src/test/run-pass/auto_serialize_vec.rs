use std;
import std::prettyprint::serializer;
import std::io;

#[auto_serialize]
type uint_vec = [uint];

fn main() {
    let ex = [1u, 2u, 3u];
    let s = io::with_str_writer {|w| uint_vec::serialize(w, ex)};
    #debug["s == %?", s];
    assert s == "[1u, 2u, 3u]";
}