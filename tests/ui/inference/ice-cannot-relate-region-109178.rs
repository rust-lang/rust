// test for ice #109178  cannot relate region: LUB(ReErased, ReError)

#![allow(incomplete_features)]
#![crate_type = "lib"]
#![feature(adt_const_params, unsized_const_params, generic_const_exprs)]

struct Changes<const CHANGES: &[&'static str]>
//~^ ERROR `&` without an explicit lifetime name cannot be used here
where
    [(); CHANGES.len()]:, {}

impl<const CHANGES: &[&str]> Changes<CHANGES> where [(); CHANGES.len()]: {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here
//~^^ ERROR `&` without an explicit lifetime name cannot be used here
