#![feature(async_closure)]
//@ only-cdb
//@ compile-flags: -g
//@ edition: 2021

// === CDB TESTS ==================================================================================

// cdb-command: g
// cdb-command: dx closure
// cdb-check:closure          [Type: coroutine_closure::main::closure_env$0]
// cdb-check:     [+0x[...]] y                : "" [Type: alloc::string::String]
// cdb-check:     [+0x[...]] x                : "" [Type: alloc::string::String]
#![allow(unused)]
fn main() {
    let x = String::new();
    let y = String::new();
    let closure = async move || {
        drop(y);
        println!("{x}");
    };

    _zzz(); // #break

    std::hint::black_box(closure);
}

#[inline(never)]
fn _zzz() {
    ()
}
