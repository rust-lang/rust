//@compile-flags: --test
//@check-pass
#![warn(clippy::missing_inline_in_public_items)]

#[expect(clippy::missing_inline_in_public_items)]
pub fn foo() -> u32 {
    0
}

fn private_function() {}
