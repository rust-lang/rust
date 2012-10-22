// xfail-fast Can't redeclare malloc with wrong signature because bugs
// Issue #3656
// Incorrect struct size computation in the FFI, because of not taking
// the alignment of elements into account.

use libc::*;

struct KEYGEN {
    hash_algorithm: [c_uint * 2],
    count: uint32_t,
    salt: *c_void,
    salt_size: uint32_t,
}

extern {
    // Bogus signature, just need to test if it compiles.
    pub fn malloc(++data: KEYGEN);
}

fn main() {
}
