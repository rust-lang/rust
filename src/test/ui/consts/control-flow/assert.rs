// Test that `assert` works only when both `const_if_match` and `const_panic` are enabled.

// revisions: stock if_match panic both

#![cfg_attr(any(both, if_match), feature(const_if_match))]
#![cfg_attr(any(both, panic), feature(const_panic))]

const _: () = assert!(true);
//[stock,panic]~^ ERROR `if` is not allowed in a `const`
//[if_match]~^^ ERROR panicking in constants is unstable

const _: () = assert!(false);
//[stock,panic]~^ ERROR `if` is not allowed in a `const`
//[if_match]~^^ ERROR panicking in constants is unstable
//[both]~^^^ ERROR any use of this value will cause an error

fn main() {}
