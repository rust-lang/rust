#![crate_type = "lib"]

macro_rules! sample { () => {} }

#[sample]           //~ ERROR cannot find attribute `sample` in this scope
#[derive(sample)]   //~ ERROR cannot find derive macro `sample` in this scope
                    //~| ERROR cannot find derive macro `sample` in this scope
                    //~| ERROR cannot find derive macro `sample` in this scope
pub struct S {}
