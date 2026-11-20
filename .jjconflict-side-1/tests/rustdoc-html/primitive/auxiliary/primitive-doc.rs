//@ compile-flags: --crate-type lib
//@ edition: 2018

#![feature(rustc_attrs)]
#![feature(no_core, lang_items)]
#![no_core]

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}

#[rustc_doc_primitive = "usize"]
/// This is the built-in type `usize`.
const _: () = ();
