#![crate_name = "oud"]
#![feature(doc_cfg, doc_cfg_hide)]

#![doc(cfg_hide(feature = "solecism"))]

// @has 'oud/struct.Solecism.html'
// @count   - '//*[@class="stab portability"]' 0
// compile-flags:--cfg feature="solecism"
#[cfg(feature = "solecism")]
pub struct Solecism;

// @has 'oud/struct.Scribacious.html'
// @count   - '//*[@class="stab portability"]' 1
// @matches - '//*[@class="stab portability"]' 'crate feature solecism'
#[cfg(feature = "solecism")]
#[doc(cfg(feature = "solecism"))]
pub struct Scribacious;

// @has 'oud/struct.Hyperdulia.html'
// @count   - '//*[@class="stab portability"]' 1
// @matches - '//*[@class="stab portability"]' 'crate feature hyperdulia'
// compile-flags:--cfg feature="hyperdulia"
#[cfg(feature = "solecism")]
#[cfg(feature = "hyperdulia")]
pub struct Hyperdulia;

// @has 'oud/struct.Oystercatcher.html'
// @count   - '//*[@class="stab portability"]' 1
// @matches - '//*[@class="stab portability"]' 'crate features solecism and oystercatcher'
// compile-flags:--cfg feature="oystercatcher"
#[cfg(all(feature = "solecism", feature = "oystercatcher"))]
pub struct Oystercatcher;
