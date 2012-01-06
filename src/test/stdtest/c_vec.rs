import core::*;

// -*- rust -*-
use std;
import std::c_vec::*;
import ctypes::*;

#[nolink]
#[abi = "cdecl"]
native mod libc {
    fn malloc(n: size_t) -> *mutable u8;
    fn free(m: *mutable u8);
}

fn malloc(n: size_t) -> t<u8> {
    let mem = libc::malloc(n);

    assert mem as int != 0;

    ret unsafe { create_with_dtor(mem, n, bind libc::free(mem)) };
}

#[test]
fn test_basic() {
    let cv = malloc(16u);

    set(cv, 3u, 8u8);
    set(cv, 4u, 9u8);
    assert get(cv, 3u) == 8u8;
    assert get(cv, 4u) == 9u8;
    assert len(cv) == 16u;
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_overrun_get() {
    let cv = malloc(16u);

    get(cv, 17u);
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn test_overrun_set() {
    let cv = malloc(16u);

    set(cv, 17u, 0u8);
}

#[test]
fn test_and_I_mean_it() {
    let cv = malloc(16u);
    let p = unsafe { ptr(cv) };

    set(cv, 0u, 32u8);
    set(cv, 1u, 33u8);
    assert unsafe { *p } == 32u8;
    set(cv, 2u, 34u8); /* safety */
}
