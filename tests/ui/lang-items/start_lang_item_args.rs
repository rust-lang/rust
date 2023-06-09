// check-fail
// revisions: missing_all_args missing_sigpipe_arg missing_ret start_ret too_many_args
// revisions: main_ty main_args main_ret argc argv_inner_ptr argv sigpipe

#![feature(lang_items, no_core)]
#![no_core]

#[lang = "copy"]
pub trait Copy {}
#[lang = "sized"]
pub trait Sized {}

#[cfg(missing_all_args)]
#[lang = "start"]
fn start<T>() -> isize {
    //[missing_all_args]~^ ERROR incorrect number of parameters
    100
}

#[cfg(missing_sigpipe_arg)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8) -> isize {
    //[missing_sigpipe_arg]~^ ERROR incorrect number of parameters
    100
}

#[cfg(missing_ret)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) {}
//[missing_ret]~^ ERROR the return type of the `start` lang item is incorrect

#[cfg(start_ret)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> u8 {
    //[start_ret]~^ ERROR the return type of the `start` lang item is incorrect
    100
}

#[cfg(too_many_args)]
#[lang = "start"]
fn start<T>(
    //[too_many_args]~^ ERROR incorrect number of parameters
    _main: fn() -> T,
    _argc: isize,
    _argv: *const *const u8,
    _sigpipe: u8,
    _extra_arg: (),
) -> isize {
    100
}

#[cfg(main_ty)]
#[lang = "start"]
fn start<T>(_main: u64, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    //[main_ty]~^ ERROR parameter 1 of the `start` lang item is incorrect
    100
}

#[cfg(main_args)]
#[lang = "start"]
fn start<T>(_main: fn(i32) -> T, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    //[main_args]~^ ERROR parameter 1 of the `start` lang item is incorrect
    100
}

#[cfg(main_ret)]
#[lang = "start"]
fn start<T>(_main: fn() -> u16, _argc: isize, _argv: *const *const u8, _sigpipe: u8) -> isize {
    //[main_ret]~^ ERROR parameter 1 of the `start` lang item is incorrect
    100
}

#[cfg(argc)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: i8, _argv: *const *const u8, _sigpipe: u8) -> isize {
    //[argc]~^ ERROR parameter 2 of the `start` lang item is incorrect
    100
}

#[cfg(argv_inner_ptr)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const usize, _sigpipe: u8) -> isize {
    //[argv_inner_ptr]~^ ERROR parameter 3 of the `start` lang item is incorrect
    100
}

#[cfg(argv)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: u8, _sigpipe: u8) -> isize {
    //[argv]~^ ERROR parameter 3 of the `start` lang item is incorrect
    100
}

#[cfg(sigpipe)]
#[lang = "start"]
fn start<T>(_main: fn() -> T, _argc: isize, _argv: *const *const u8, _sigpipe: i64) -> isize {
    //[sigpipe]~^ ERROR parameter 4 of the `start` lang item is incorrect
    100
}

fn main() {}
