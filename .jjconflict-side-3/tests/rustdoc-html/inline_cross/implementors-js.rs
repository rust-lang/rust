//@ aux-build:implementors_inline.rs
//@ build-aux-docs
//@ ignore-cross-compile

extern crate implementors_inline;

//@ !has trait.impl/implementors_js/trait.MyTrait.js
//@ has trait.impl/implementors_inline/my_trait/trait.MyTrait.js
//@ !has trait.impl/implementors_inline/prelude/trait.MyTrait.js
//@ has implementors_inline/my_trait/trait.MyTrait.html
//@ has - '//script/@src' '../../trait.impl/implementors_inline/my_trait/trait.MyTrait.js'
//@ has implementors_js/trait.MyTrait.html
//@ has - '//script/@src' '../trait.impl/implementors_inline/my_trait/trait.MyTrait.js'
/// When re-exporting this trait, the HTML will be inlined,
/// but, vitally, the JavaScript will be located only at the
/// one canonical path.
pub use implementors_inline::prelude::MyTrait;

pub struct OtherStruct;

impl MyTrait for OtherStruct {
    fn my_fn(&self) -> OtherStruct {
        OtherStruct
    }
}
