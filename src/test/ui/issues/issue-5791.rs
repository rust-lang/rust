// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

extern {
    #[link_name = "malloc"]
    fn malloc1(len: i32) -> *const u8;
    #[link_name = "malloc"]
    fn malloc2(len: i32, foo: i32) -> *const u8;
}

pub fn main () {}
