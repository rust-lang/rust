use std;

import std::vec;
import std::str;

native "cdecl" mod libc = "" {
    fn my_strlen(str: *u8) -> uint = "strlen";
}

fn strlen(str: str) -> uint unsafe {
    // C string is terminated with a zero
    let bytes = str::bytes(str) + [0u8];
    ret libc::my_strlen(vec::unsafe::to_ptr(bytes));
}

fn main() {
    let len = strlen("Rust");
    assert(len == 4u);
}
