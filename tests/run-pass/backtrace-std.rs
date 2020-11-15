// normalize-stderr-test "at .*/(rust[^/]*|checkout)/library/" -> "at RUSTLIB/"
// normalize-stderr-test "RUSTLIB/([^:]*):\d+:\d+"-> "RUSTLIB/$1:LL:CC"
// normalize-stderr-test "::<.*>" -> ""
// compile-flags: -Zmiri-disable-isolation

#![feature(backtrace)]

use std::backtrace::Backtrace;

#[inline(never)] fn func_a() -> Backtrace { func_b::<u8>() }
#[inline(never)] fn func_b<T>() -> Backtrace { func_c() }

macro_rules! invoke_func_d {
    () => { func_d() }
}

#[inline(never)] fn func_c() -> Backtrace { invoke_func_d!() }
#[inline(never)] fn func_d() -> Backtrace { Backtrace::capture() }

fn main() {
    eprint!("{}", func_a());
}
