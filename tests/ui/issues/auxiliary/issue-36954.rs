#![crate_type = "lib"]

const fn foo(i: i32) -> i32 {
    i
}

pub const FOO: i32 = foo(1);
