#![feature(lint_reasons)]

#![forbid(
    unsafe_code,
    //~^ NOTE `forbid` level set here
    reason = "our errors & omissions insurance policy doesn't cover unsafe Rust"
)]

use std::ptr;

fn main() {
    let a_billion_dollar_mistake = ptr::null();

    #[allow(unsafe_code)]
    //~^ ERROR allow(unsafe_code) overruled by outer forbid(unsafe_code)
    //~| NOTE overruled by previous forbid
    //~| NOTE our errors & omissions insurance policy doesn't cover unsafe Rust
    unsafe {
        *a_billion_dollar_mistake
    }
}
