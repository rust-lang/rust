// edition:2018

#![no_core]
#![feature(no_core)]

// @count macro.json "$.index[*][?(@.name=='macro')].inner.items[*]" 2

// @set repro_id = macro.json "$.index[*][?(@.name=='repro')].id"
// @has - "$.index[*][?(@.name=='macro')].inner.items[*]" $repro_id
#[macro_export]
macro_rules! repro {
    () => {};
}

// @set repro2_id = macro.json "$.index[*][?(@.inner.name=='repro2')].id"
// @has - "$.index[*][?(@.name=='macro')].inner.items[*]" $repro2_id
pub use crate::repro as repro2;
