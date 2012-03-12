use std;
import std::prettyprint::serializer;
import std::io;

// Test where we link various types used by name.

#[auto_serialize]
type spanned<T> = {lo: uint, hi: uint, node: T};

#[auto_serialize]
type spanned_uint = spanned<uint>;

fn main() {
    let x: spanned_uint = {lo: 0u, hi: 5u, node: 22u};
    spanned_uint::serialize(io::stdout(), x);
    let s = io::with_str_writer {|w| spanned_uint::serialize(w, x)};
    #debug["s == %?", s];
    assert s == "{lo: 0u, hi: 5u, node: 22u}";
}