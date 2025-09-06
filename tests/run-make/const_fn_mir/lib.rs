// emit-mir
//@ check-pass

#![crate_type = "lib"]

pub const fn foo() -> i32 {
    5 + 6
}
