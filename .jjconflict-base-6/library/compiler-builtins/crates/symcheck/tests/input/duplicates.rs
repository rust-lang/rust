//! Ensure we catch duplicate symbols (the duplicates are in the aux file). Gets built twice
//! as separate object files.

#![no_std]

#[unsafe(no_mangle)]
static IDUP: i32 = 0;
#[unsafe(no_mangle)]
static FDUP: f32 = 0.0;

#[unsafe(no_mangle)]
pub fn fndup() {}
