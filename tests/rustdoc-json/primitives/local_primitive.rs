// Regression test for <https://github.com/rust-lang/rust/issues/104064>.

#![feature(no_core)]
#![feature(rustc_attrs)]
#![feature(rustdoc_internals)]
#![feature(lang_items)]
#![no_core]
#![rustc_coherence_is_core]

//! Link to [i32][prim@i32] [i64][prim@i64]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[rustc_doc_primitive = "i32"]
const _: () = ();

//@ set local_i32 = "$.index[?(@.name=='i32')].id"

//@ has "$.index[?(@.name=='local_primitive')]"
//@ is "$.index[?(@.name=='local_primitive')].links['prim@i32']" $local_i32
