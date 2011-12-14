use std;

import vec;
import str;

#[link_name = ""]
#[abi = "cdecl"]
native mod libc {
    #[link_name = "strlen"]
    fn my_strlen(str: *u8) -> uint;
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
