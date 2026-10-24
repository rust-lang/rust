#![feature(pattern_types, pattern_type_macro)]
#![crate_name = "pattern_types_implementors"]

use std::pat::pattern_type;

pub trait MyTrait {}

impl MyTrait for pattern_type!(*const u8 is !null) {}

//@ has pattern_types_implementors/trait.MyTrait.html
//@ has - '//*[@id="implementors-list"]/*[@class="impl"]' 'impl MyTrait for *const u8 is !null'
//@ !has - '//*[@id="implementors-list"]/*[@class="impl"]' 'TyPat {'
