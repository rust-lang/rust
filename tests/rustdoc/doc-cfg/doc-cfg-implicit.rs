#![crate_name = "funambulism"]
#![feature(doc_cfg)]

//@ has 'funambulism/struct.Disorbed.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate feature disorbed'
//@ compile-flags:--cfg feature="disorbed"
#[cfg(feature = "disorbed")]
pub struct Disorbed;

//@ has 'funambulism/struct.Aesthesia.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate feature aesthesia'
//@ compile-flags:--cfg feature="aesthesia"
#[doc(cfg(feature = "aesthesia"))]
pub struct Aesthesia;

//@ has 'funambulism/struct.Pliothermic.html'
//@ count   - '//*[@class="stab portability"]' 1
//@ matches - '//*[@class="stab portability"]' 'crate feature pliothermic'
//@ compile-flags:--cfg feature="epopoeist"
#[cfg(feature = "epopoeist")]
#[doc(cfg(feature = "pliothermic"))]
pub struct Pliothermic;

//@ has 'funambulism/struct.Simillimum.html'
//@ count   - '//*[@class="stab portability"]' 0
//@ compile-flags:--cfg feature="simillimum"
#[cfg(feature = "simillimum")]
#[doc(cfg(all()))]
pub struct Simillimum;
