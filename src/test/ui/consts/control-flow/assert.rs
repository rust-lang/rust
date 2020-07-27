// Test that `assert` works when `const_panic` is enabled.

// revisions: stock panic

#![cfg_attr(panic, feature(const_panic))]

const _: () = assert!(true);
//[stock]~^ ERROR panicking in constants is unstable

const _: () = assert!(false);
//[stock]~^ ERROR panicking in constants is unstable
//[panic]~^^ ERROR any use of this value will cause an error

fn main() {}
