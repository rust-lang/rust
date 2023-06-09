//@compile-flags: -Zmiri-disable-isolation
//@rustc-env: RUST_BACKTRACE=1

use std::backtrace::Backtrace;

#[inline(never)]
fn func_a() -> Backtrace {
    func_b::<u8>()
}
#[inline(never)]
fn func_b<T>() -> Backtrace {
    func_c()
}

macro_rules! invoke_func_d {
    () => {
        func_d()
    };
}

#[inline(never)]
fn func_c() -> Backtrace {
    invoke_func_d!()
}
#[inline(never)]
fn func_d() -> Backtrace {
    Backtrace::capture()
}

fn main() {
    eprint!("{}", func_a());
}
