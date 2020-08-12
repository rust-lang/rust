// Regression test for https://github.com/rust-lang/rust/issues/56445#issuecomment-518402995.
// revisions: full min
#![cfg_attr(full, feature(const_generics))] //[full]~WARN the feature `const_generics` is incomplete
#![cfg_attr(min, feature(min_const_generics))]
#![crate_type = "lib"]

use std::marker::PhantomData;

struct Bug<'a, const S: &'a str>(PhantomData<&'a ()>);
//~^ ERROR: use of non-static lifetime `'a` in const generic
//[min]~| ERROR: using `&'static str` as const

impl Bug<'_, ""> {}
