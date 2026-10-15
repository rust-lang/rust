//@ ignore-windows
//@ compile-flags: --remap-path-prefix={{src-base}}=/remapped
//@ compile-flags: --remap-path-scope=documentation -Zunstable-options

//@ is "$.index[?(@.name=='StrLike')].span.filename" \"/remapped/remap-path-prefix/doc.rs\"
pub trait StrLike {}
