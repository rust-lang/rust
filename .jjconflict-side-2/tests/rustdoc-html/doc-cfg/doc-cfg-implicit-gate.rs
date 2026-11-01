//@ compile-flags:--cfg feature="worricow"
#![feature(doc_cfg)]
#![crate_name = "xenogenous"]

//@ has 'xenogenous/struct.Worricow.html'
//@ count   - '//*[@class="stab portability"]' 1
#[cfg(feature = "worricow")]
pub struct Worricow;
