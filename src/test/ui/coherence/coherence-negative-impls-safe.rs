// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]

use std::marker::Send;

struct TestType;

unsafe impl !Send for TestType {}
//[old]~^ ERROR negative impls cannot be unsafe
//[re]~^^ ERROR E0198

fn main() {}
