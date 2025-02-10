#![crate_type = "lib"]

#[core::contracts::requires(x > 0)]
pub fn requires_needs_it(x: i32) { }
//~^^  ERROR use of unstable library feature `contracts`
//~^^^ ERROR contracts are incomplete

#[core::contracts::ensures(|ret| *ret > 0)]
pub fn ensures_needs_it() -> i32 { 10 }
//~^^  ERROR use of unstable library feature `contracts`
//~^^^ ERROR contracts are incomplete
