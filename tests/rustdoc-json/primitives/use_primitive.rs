//@ edition:2018

#![feature(rustc_attrs)]

#[rustc_doc_primitive = "usize"]
mod usize {}

//@ set local_crate_id = "$.index[?(@.name=='use_primitive')].crate_id"

//@ has "$.index[?(@.name=='ilog10')]"
//@ !is "$.index[?(@.name=='ilog10')].crate_id" $local_crate_id
//@ has "$.index[?(@.name=='checked_add')]"
//@ !is "$.index[?(@.name=='checked_add')]" $local_crate_id
//@ !has "$.index[?(@.name=='is_ascii_uppercase')]"

//@ is "$.index[?(@.inner.use.name=='my_i32')].inner.use.id" null
pub use i32 as my_i32;
//@ is "$.index[?(@.inner.use.name=='u32')].inner.use.id" null
pub use u32;
