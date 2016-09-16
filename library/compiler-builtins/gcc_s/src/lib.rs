#![feature(drop_types_in_const)]

extern crate libloading;

use std::sync::{Once, ONCE_INIT};

use libloading::Library;

static mut GCC_S: Option<Library> = None;

fn gcc_s() -> &'static Library {
    unsafe {
        static INIT: Once = ONCE_INIT;

        INIT.call_once(|| {
            GCC_S = Some(Library::new("libgcc_s.so.1").unwrap());
        });
        GCC_S.as_ref().unwrap()
    }
}

macro_rules! declare {
    ($symbol:ident: fn($($i:ty),+) -> $o:ty) => {
        pub fn $symbol() -> Option<unsafe extern fn($($i),+) -> $o> {
            unsafe {
                gcc_s().get(concat!("__", stringify!($symbol)).as_bytes()).ok().map(|s| *s)
            }
        }
    }
}

declare!(ashldi3: fn(u64, u32) -> u64);
declare!(ashrdi3: fn(i64, u32) -> i64);
declare!(divdi3: fn(i64, i64) -> i64);
declare!(divmoddi4: fn(i64, i64, &mut i64) -> i64);
declare!(divmodsi4: fn(i32, i32, &mut i32) -> i32);
declare!(divsi3: fn(i32, i32) -> i32);
declare!(lshrdi3: fn(u64, u32) -> u64);
declare!(moddi3: fn(i64, i64) -> i64);
declare!(modsi3: fn(i32, i32) -> i32);
declare!(muldi3: fn(u64, u64) -> u64);
declare!(mulodi4: fn(i64, i64, &mut i32) -> i64);
declare!(mulosi4: fn(i32, i32, &mut i32) -> i32);
declare!(udivdi3: fn(u64, u64) -> u64);
declare!(udivmoddi4: fn(u64, u64, Option<&mut u64>) -> u64);
declare!(udivmodsi4: fn(u32, u32, Option<&mut u32>) -> u32);
declare!(udivsi3: fn(u32, u32) -> u32);
declare!(umoddi3: fn(u64, u64) -> u64);
declare!(umodsi3: fn(u32, u32) -> u32);
declare!(addsf3: fn(f32, f32) -> f32);
declare!(adddf3: fn(f64, f64) -> f64);
