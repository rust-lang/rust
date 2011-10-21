// xfail-test

import std::sys;

// The purpose of this test is to check that we can
// successfully (and safely) invoke external, c-stack-cdecl
// functions from outside the crate.

fn main() {
    let foo = sys::rustrt::last_os_error();
}