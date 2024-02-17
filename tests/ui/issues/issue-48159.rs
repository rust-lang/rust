//@ run-pass
#![allow(non_camel_case_types)]

use std::mem;

pub enum c_void {}

type uintptr_t = usize;
type int16_t = u16;
type uint16_t = int16_t;
type uint32_t = u32;
type intptr_t = uintptr_t;

#[repr(C)]
#[repr(packed(4))]
pub struct kevent {
    pub ident: uintptr_t,
    pub filter: int16_t,
    pub flags: uint16_t,
    pub fflags: uint32_t,
    pub data: intptr_t,
    pub udata: *mut c_void,
}

fn main() {
    assert_eq!(mem::align_of::<kevent>(), 4);
}
