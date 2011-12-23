import core::*;

import sys;

#[test]
fn last_os_error() {
    log(debug, sys::last_os_error());
}

#[test]
fn size_of_basic() {
    assert sys::size_of::<u8>() == 1u;
    assert sys::size_of::<u16>() == 2u;
    assert sys::size_of::<u32>() == 4u;
    assert sys::size_of::<u64>() == 8u;
}

#[test]
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
fn size_of_32() {
    assert sys::size_of::<uint>() == 4u;
    assert sys::size_of::<*uint>() == 4u;
}

#[test]
#[cfg(target_arch = "x86_64")]
fn size_of_64() {
    assert sys::size_of::<uint>() == 8u;
    assert sys::size_of::<*uint>() == 8u;
}

#[test]
fn align_of_basic() {
    assert sys::align_of::<u8>() == 1u;
    assert sys::align_of::<u16>() == 2u;
    assert sys::align_of::<u32>() == 4u;
}

#[test]
#[cfg(target_arch = "x86")]
#[cfg(target_arch = "arm")]
fn align_of_32() {
    assert sys::align_of::<uint>() == 4u;
    assert sys::align_of::<*uint>() == 4u;
}

#[test]
#[cfg(target_arch = "x86_64")]
fn align_of_64() {
    assert sys::align_of::<uint>() == 8u;
    assert sys::align_of::<*uint>() == 8u;
}