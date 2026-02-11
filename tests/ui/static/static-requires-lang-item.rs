//! Test that creating statics requires the `AllowSharedStatic` language item.
#![feature(lang_items, no_core)]
#![no_core]
#![no_main]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
pub trait Sized: MetaSized {}
#[lang = "copy"]
trait Copy {}

static FOO: () = ();
//~^ ERROR: requires `allow_shared_static` lang_item
