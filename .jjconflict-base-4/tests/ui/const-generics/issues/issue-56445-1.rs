// Regression test for https://github.com/rust-lang/rust/issues/56445#issuecomment-518402995.
//@ revisions: full min
#![cfg_attr(full, feature(adt_const_params, unsized_const_params))]
#![cfg_attr(full, allow(incomplete_features))]
#![crate_type = "lib"]

use std::marker::PhantomData;

struct Bug<'a, const S: &'a str>(PhantomData<&'a ()>);
//~^ ERROR: the type of const parameters must not depend on other generic parameters

impl Bug<'_, ""> {}
