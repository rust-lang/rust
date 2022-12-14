// edition:2018

#![no_core]
#![feature(no_core)]

// @set repro_id = "$.index[*][?(@.name=='repro')].id"
#[macro_export]
macro_rules! repro {
    () => {};
}

// @set repro2_id = "$.index[*][?(@.inner.name=='repro2')].id"
pub use crate::repro as repro2;

// @ismany "$.index[*][?(@.name=='macro')].inner.items[*]" $repro_id $repro2_id
