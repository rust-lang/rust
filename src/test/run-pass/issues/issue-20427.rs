// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]

// aux-build:i8.rs
// ignore-pretty issue #37201

extern crate i8;
use std::string as i16;
static i32: i32 = 0;
const i64: i64 = 0;
fn u8(f32: f32) {}
fn f<f64>(f64: f64) {}
enum u32 {}
struct u64;
trait bool {}

mod char {
    extern crate i8;
    static i32_: i32 = 0;
    const i64_: i64 = 0;
    fn u8_(f32: f32) {}
    fn f_<f64_>(f64: f64_) {}
    type u16_ = u16;
    enum u32_ {}
    struct u64_;
    trait bool_ {}
    mod char_ {}

    mod str {
        use super::i8 as i8;
        use super::i32_ as i32;
        use super::i64_ as i64;
        use super::u8_ as u8;
        use super::f_ as f64;
        use super::u16_ as u16;
        use super::u32_ as u32;
        use super::u64_ as u64;
        use super::bool_ as bool;
        use super::{bool_ as str};
        use super::char_ as char;
    }
}

trait isize_ {
    type isize;
}

fn usize<'usize>(usize: &'usize usize) -> &'usize usize { usize }

mod reuse {
    use std::mem::size_of;

    type u8 = u64;
    use std::string::String as i16;

    pub fn check<u16>() {
        assert_eq!(size_of::<u8>(), 8);
        assert_eq!(size_of::<::u64>(), 0);
        assert_eq!(size_of::<i16>(), 3 * size_of::<*const ()>());
        assert_eq!(size_of::<u16>(), 0);
    }
}

mod guard {
    pub fn check() {
        use std::u8; // bring module u8 in scope
        fn f() -> u8 { // OK, resolves to primitive u8, not to std::u8
            u8::max_value() // OK, resolves to associated function <u8>::max_value,
                            // not to non-existent std::u8::max_value
        }
        assert_eq!(f(), u8::MAX); // OK, resolves to std::u8::MAX
    }
}

fn main() {
    let bool = true;
    let _ = match bool {
        str @ true => if str { i32 as i64 } else { i64 },
        false => i64,
    };

    reuse::check::<u64>();
    guard::check();
}
