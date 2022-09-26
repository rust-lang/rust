// compile-flags:--cfg feature="worricow"
#![crate_name = "xenogenous"]

#![doc(auto_cfg)]

// @has 'xenogenous/struct.Worricow.html'
// @count   - '//*[@class="stab portability"]' 1
#[cfg(feature = "worricow")]
pub struct Worricow;
