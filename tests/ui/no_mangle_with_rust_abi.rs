#![allow(unused)]
#![warn(clippy::no_mangle_with_rust_abi)]

#[no_mangle]
fn rust_abi_fn_one(arg_one: u32, arg_two: usize) {}

#[no_mangle]
pub fn rust_abi_fn_two(arg_one: u32, arg_two: usize) {}

/// # Safety
/// This function shouldn't be called unless the horsemen are ready
#[no_mangle]
pub unsafe fn rust_abi_fn_three(arg_one: u32, arg_two: usize) {}

/// # Safety
/// This function shouldn't be called unless the horsemen are ready
#[no_mangle]
unsafe fn rust_abi_fn_four(arg_one: u32, arg_two: usize) {}

#[no_mangle]
fn rust_abi_multiline_function_really_long_name_to_overflow_args_to_multiple_lines(
    arg_one: u32,
    arg_two: usize,
) -> u32 {
    0
}

// Must not run on functions that explicitly opt in to Rust ABI with `extern "Rust"`
#[no_mangle]
#[rustfmt::skip]
extern "Rust" fn rust_abi_fn_explicit_opt_in(arg_one: u32, arg_two: usize) {}

fn rust_abi_fn_again(arg_one: u32, arg_two: usize) {}

#[no_mangle]
extern "C" fn c_abi_fn(arg_one: u32, arg_two: usize) {}

extern "C" fn c_abi_fn_again(arg_one: u32, arg_two: usize) {}

extern "C" {
    fn c_abi_in_block(arg_one: u32, arg_two: usize);
}

fn main() {
    // test code goes here
}
