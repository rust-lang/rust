// Provides a safe way to get C pointers to strings, because in many
// places LLVM wants a string but doesn't take a copy.

import std::str;

export t;
export mk;
export get_cstr;

type t_ = @{
    // This string is boxed so that I remember that it has to be boxed
    // after the ivec conversion.
    mutable cache: [@str]
};

tag t {
    private(t_);
}

fn mk() -> t {
    ret private(@{mutable cache: []});
}

fn get_cstr(t: &t, s: &str) -> str::rustrt::sbuf {
    let boxed = @s;
    let buf = str::buf(*boxed);
    (*t).cache += [boxed];
    ret buf;
}