// compile-flags:--cfg feature="worricow"
#![crate_name = "xenogenous"]

// @has 'xenogenous/struct.Worricow.html'
// @count   - '//*[@class="stab portability"]' 0
#[cfg(feature = "worricow")]
pub struct Worricow;
