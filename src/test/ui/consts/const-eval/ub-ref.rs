// ignore-tidy-linelength
#![feature(const_transmute)]
#![allow(const_err, invalid_value)] // make sure we cannot allow away the errors tested here

use std::mem;

const UNALIGNED: &u16 = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR it is undefined behavior to use this value
//~| type validation failed: encountered an unaligned reference (required 2 byte alignment but found 1)

const UNALIGNED_BOX: Box<u16> = unsafe { mem::transmute(&[0u8; 4]) };
//~^ ERROR it is undefined behavior to use this value
//~| type validation failed: encountered an unaligned box (required 2 byte alignment but found 1)

const NULL: &u16 = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

const NULL_BOX: Box<u16> = unsafe { mem::transmute(0usize) };
//~^ ERROR it is undefined behavior to use this value

// It is very important that we reject this: We do promote `&(4 * REF_AS_USIZE)`,
// but that would fail to compile; so we ended up breaking user code that would
// have worked fine had we not promoted.
const REF_AS_USIZE: usize = unsafe { mem::transmute(&0) };
//~^ ERROR it is undefined behavior to use this value

const REF_AS_USIZE_SLICE: &[usize] = &[unsafe { mem::transmute(&0) }];
//~^ ERROR it is undefined behavior to use this value

const REF_AS_USIZE_BOX_SLICE: Box<[usize]> = unsafe { mem::transmute::<&[usize], _>(&[mem::transmute(&0)]) };
//~^ ERROR it is undefined behavior to use this value

const USIZE_AS_REF: &'static u8 = unsafe { mem::transmute(1337usize) };
//~^ ERROR it is undefined behavior to use this value

const USIZE_AS_BOX: Box<u8> = unsafe { mem::transmute(1337usize) };
//~^ ERROR it is undefined behavior to use this value

fn main() {}
