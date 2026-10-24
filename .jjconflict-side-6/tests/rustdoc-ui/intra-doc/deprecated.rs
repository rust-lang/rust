#![deny(rustdoc::broken_intra_doc_links)]

#[deprecated = "[broken cross-reference](TypeAlias::hoge)"] //~ ERROR
pub struct A;

#[deprecated(since = "0.0.0", note = "[broken cross-reference](TypeAlias::hoge)")] //~ ERROR
pub struct B1;

#[deprecated(note = "[broken cross-reference](TypeAlias::hoge)", since = "0.0.0")] //~ ERROR
pub struct B2;
