//@ ignore-windows
//@ compile-flags: --remap-path-prefix={{src-base}}=/remapped
//@ compile-flags: --remap-path-scope=object -Zunstable-options

//@ is "$.index[?(@.name=='StrLike')].span.filename" \"/remapped/remap-path-prefix/object.rs\"
pub trait StrLike {}
