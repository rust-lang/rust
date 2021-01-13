// run-pass
#![allow(dead_code)]
#![warn(clashing_extern_declarations)]
// pretty-expanded FIXME #23616

extern "C" {
    #[link_name = "malloc"]
    fn malloc1(len: i32) -> *const u8;
    #[link_name = "malloc"]
    //~^ WARN `malloc2` redeclares `malloc` with a different signature
    fn malloc2(len: i32, foo: i32) -> *const u8;
}

pub fn main() {}
