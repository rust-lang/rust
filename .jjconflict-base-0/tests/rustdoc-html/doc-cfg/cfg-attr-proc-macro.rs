//@ aux-build: cfg-attr-proc-macro.rs

#![crate_name = "foo"]
#![feature(doc_cfg)]

extern crate cfg_attr_proc_macro;

pub trait Trait {}

//@ has 'foo/struct.B.html'
//@ has - '//*[@id="impl-Trait-for-B"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on non-crate feature boop only.'
#[cfg_attr(not(feature = "boop"), derive(cfg_attr_proc_macro::Yop))]
pub struct B;
