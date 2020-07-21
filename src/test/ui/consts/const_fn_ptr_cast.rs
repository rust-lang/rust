#![feature(const_fn)]
#![feature(const_fn_pointer)]
#[allow(dead_code)]

const fn const_fn() { }
fn not_const_fn() { }
unsafe fn unsafe_fn() { }
const unsafe fn const_unsafe_fn() { }

const _: fn() = const_fn;
const _: unsafe fn() = const_fn;
const _: const unsafe fn() = const_fn;

const _: const fn() = not_const_fn;
//~^ ERROR mismatched types

const _: const fn() = unsafe_fn;
//~^ ERROR mismatched types

const _: const unsafe fn() = unsafe_fn;
//~^ ERROR mismatched types

const _: unsafe fn() = const_unsafe_fn;

fn main() { }