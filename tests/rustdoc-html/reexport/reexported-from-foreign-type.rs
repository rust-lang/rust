#![feature(extern_types)]
#![feature(lang_items)]
#![feature(no_core)]
#![crate_name = "foo"]

#![no_std]
#![no_core]

pub mod a {
    unsafe extern "C" {
        #[lang = "code"]
        pub type Bib;
    }
}

//@ has 'foo/foreigntype.Bib.html'
//@ has - '//*[@class="reexport-origin"]' 'Reexported from crate::a::Bib'
//@ has - '//*[@class="reexport-origin"]/a[@href="a/foreigntype.Bib.html"]' 'crate::a::Bib'
#[doc(inline)]
pub use crate::a::Bib;
