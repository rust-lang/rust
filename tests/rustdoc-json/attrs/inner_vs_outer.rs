#![no_std]

//@ is "$.index[?(@.name=='foo')].attrs" '[{"content": "#[allow(unused)]", "is_inner": false}, {"content": "#[allow(dead_code)]", "is_inner": true}]'
#[allow(unused)]
pub mod foo {
    #![allow(dead_code)]
}
