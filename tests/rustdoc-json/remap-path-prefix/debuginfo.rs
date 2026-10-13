//@ compile-flags: --remap-path-prefix={{src-base}}=/remapped
//@ compile-flags: --remap-path-scope=debuginfo -Zunstable-options

//@ is "$.index[?(@.name=='StrLike')].span.filename" $FILE
pub trait StrLike {}
