// build-fail

#![feature(lang_items, no_core)]
#![no_core]

#[lang="copy"] pub trait Copy { }
#[lang="sized"] pub trait Sized { }

//@error-in-other-file:requires `start` lang_item

fn main() {}
