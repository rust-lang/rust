#![crate_name = "oud"]
#![feature(doc_cfg)]

#![doc(auto_cfg(hide(feature = "solecism")))]

//@ has 'oud/struct.Solecism.html'
//@ count   - '//*[@class="stab portability"]' 0
//@ compile-flags:--cfg feature="solecism"
#[cfg(feature = "solecism")]
pub struct Solecism;

//@ has 'oud/struct.Scribacious.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate feature solecism'
#[cfg(feature = "solecism")]
#[doc(cfg(feature = "solecism"))]
pub struct Scribacious;

//@ has 'oud/struct.Hyperdulia.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate features hyperdulia only'
//@ compile-flags:--cfg feature="hyperdulia"
#[cfg(feature = "solecism")]
#[cfg(feature = "hyperdulia")]
pub struct Hyperdulia;

//@ has 'oud/struct.Oystercatcher.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate features oystercatcher only'
//@ compile-flags:--cfg feature="oystercatcher"
#[cfg(all(feature = "solecism", feature = "oystercatcher"))]
pub struct Oystercatcher;
