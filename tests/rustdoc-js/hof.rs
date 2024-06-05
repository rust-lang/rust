#![feature(never_type)]

pub struct First<T>(T);
pub struct Second<T>(T);
pub struct Third<T>(T);

pub fn fn_ptr(_: fn(First<u32>) -> !, _: bool) {}
pub fn fn_once(_: impl FnOnce(Second<u32>) -> !, _: u8) {}
pub fn fn_mut(_: impl FnMut(Third<u32>) -> !, _: i8) {}
pub fn fn_(_: impl Fn(u32) -> !, _: char) {}

pub fn multiple(_: impl Fn(&'static str, &'static str) -> i8) {}
