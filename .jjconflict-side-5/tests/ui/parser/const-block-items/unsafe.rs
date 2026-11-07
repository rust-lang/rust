#![feature(const_block_items)]

const unsafe fn foo() -> bool {
    true
}

const unsafe { assert!(foo()) }
//~^ ERROR: expected one of `extern` or `fn`, found `{`

fn main() { }
