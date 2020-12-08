// Verify that we do not warn on types that are used by foreign functions.
// check-pass
#![deny(dead_code)]

#[repr(C)]
struct Type(u8);

#[repr(C)]
struct Param(u8);

extern "C" {
    #[allow(dead_code)]
    fn hey(t: Param);

    #[allow(dead_code)]
    static much: Type;
}

fn main() {}
