#![feature(start)]

#[start]
fn foo(argc: isize, argv: *const *const u8) -> isize { 0 }

#[start]
fn f(argc: isize, argv: *const *const u8) -> isize { 0 }
//~^ ERROR E0138
