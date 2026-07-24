// We are core.
#![feature(lang_items, no_core)]
#![allow(internal_features)]
#![no_std]
#![no_core]
#![crate_type = "rlib"]

// Hack: this is needed for the import prelude resolution.
pub mod prelude {
    pub mod rust_2024 {}
}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[lang = "sized"]
pub trait Sized: MetaSized {}
