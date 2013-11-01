// Any copyright is dedicated to the Public Domain.
// http://creativecommons.org/publicdomain/zero/1.0/

// pp-exact

fn call_it(f: proc(~str) -> ~str) { }

fn call_this(f: |&str|: Send) { }

fn call_that(f: <'a>|&'a int, &'a int|: -> int) { }

fn call_extern(f: fn() -> int) { }

fn call_abid_extern(f: extern "C" fn() -> int) { }

pub fn main() { }

