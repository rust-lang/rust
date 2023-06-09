// Any copyright is dedicated to the Public Domain.
// http://creativecommons.org/publicdomain/zero/1.0/

// pp-exact

fn call_it(f: Box<FnMut(String) -> String>) {}

fn call_this<F>(f: F) where F: Fn(&str) + Send {}

fn call_that<F>(f: F) where F: for<'a> Fn(&'a isize, &'a isize) -> isize {}

fn call_extern(f: fn() -> isize) {}

fn call_abid_extern(f: extern "C" fn() -> isize) {}

pub fn main() {}
