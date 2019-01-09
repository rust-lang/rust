// revisions: old re

#![cfg_attr(re, feature(re_rebalance_coherence))]
#![feature(optin_builtin_traits)]

auto trait MySafeTrait {}

struct Foo;

unsafe impl MySafeTrait for Foo {}
//[old]~^ ERROR implementing the trait `MySafeTrait` is not unsafe
//[re]~^^ ERROR E0199

unsafe auto trait MyUnsafeTrait {}

impl MyUnsafeTrait for Foo {}
//[old]~^ ERROR the trait `MyUnsafeTrait` requires an `unsafe impl` declaration
//[re]~^^ ERROR E0200

fn main() {}
