#![feature(lint_reasons)]

// If you turn off deduplicate diagnostics (which rustc turns on by default but
// compiletest turns off when it runs ui tests), then the errors are
// (unfortunately) repeated here because the checking is done as we read in the
// errors, and currently that happens two or three different times, depending on
// compiler flags.
//
// The test is much cleaner if we deduplicate, though.

// compile-flags: -Z deduplicate-diagnostics=true

#![forbid(
    unsafe_code,
    //~^ NOTE `forbid` level set here
    //~| NOTE the lint level is defined here
    reason = "our errors & omissions insurance policy doesn't cover unsafe Rust"
)]

use std::ptr;

fn main() {
    let a_billion_dollar_mistake = ptr::null();

    #[allow(unsafe_code)]
    //~^ ERROR allow(unsafe_code) incompatible with previous forbid
    //~| NOTE our errors & omissions insurance policy doesn't cover unsafe Rust
    //~| NOTE overruled by previous forbid
    unsafe {
        //~^ ERROR usage of an `unsafe` block
        //~| NOTE our errors & omissions insurance policy doesn't cover unsafe Rust
        *a_billion_dollar_mistake
    }
}
