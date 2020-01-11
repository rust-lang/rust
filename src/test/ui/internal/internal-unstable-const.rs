#![feature(staged_api)]
#![feature(const_if_match)]

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
const fn foo() -> i32 {
    if true { 4 } else { 5 } //~ loops and conditional expressions are not stable in const fn
}

fn main() {}
