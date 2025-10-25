// Regression test for issue #145514
// compile-flags: -C debuginfo=2
//@ build-pass

#![feature(pattern_types, pattern_type_macro, pattern_type_range_trait)]

use std::io::{Write, stderr};
use std::pat;

type SmallInt = pat::pattern_type!(i8 is 1..=2);

fn main() {
    let s: SmallInt = unsafe { core::mem::transmute(1i8) };
    let _ = stderr().lock().write_fmt(format_args!("{}\n", unsafe {
        core::mem::transmute::<_, i8>(s)
    }));
}
