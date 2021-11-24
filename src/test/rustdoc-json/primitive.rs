// edition:2018

#![feature(rustdoc_internals)]

#[doc(primitive = "usize")]
mod usize {}

// @set local_crate_id = primitive.json "$.index[*][?(@.name=='primitive')].crate_id"

// @has - "$.index[*][?(@.name=='log10')]"
// @!is - "$.index[*][?(@.name=='log10')].crate_id" $local_crate_id
// @has - "$.index[*][?(@.name=='checked_add')]"
// @!is - "$.index[*][?(@.name=='checked_add')]" $local_crate_id
// @!has - "$.index[*][?(@.name=='is_ascii_uppercase')]"
