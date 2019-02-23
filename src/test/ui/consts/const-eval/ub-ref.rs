// ignore-tidy-linelength
#![feature(const_transmute)]
#![allow(const_err)] // make sure we cannot allow away the errors tested here

use std::mem;

const UNALIGNED: &u16 = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR it is undefined behavior to use this value
//~^^ type validation failed: encountered unaligned reference (required 2 byte alignment but found 1)

const NULL: &u16 = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

const REF_AS_USIZE: usize = unsafe { mem::transmute(&0) };
//~^ ERROR it is undefined behavior to use this value

const REF_AS_USIZE_SLICE: &[usize] = &[unsafe { mem::transmute(&0) }];
//~^ ERROR it is undefined behavior to use this value

const USIZE_AS_REF: &'static u8 = unsafe { mem::transmute(1337usize) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
