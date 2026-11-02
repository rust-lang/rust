//@ ignore-windows
//@ compile-flags: --remap-path-prefix={{src-base}}=/remapped

//@ is "$.index[?(@.name=='StrLike')].span.filename" \"/remapped/remap-path-prefix/remap-path-prefix.rs\"
pub trait StrLike {}
