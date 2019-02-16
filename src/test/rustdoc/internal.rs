// compile-flags: -Z force-unstable-if-unmarked

// @matches internal/index.html '//*[@class="docblock-short"]/span[@class="stab internal"]' \
//      'Internal'
// @matches - '//*[@class="docblock-short"]' 'Docs'

// @has internal/struct.S.html '//*[@class="stab internal"]' \
//      'This is an internal compiler API. (rustc_private)'
/// Docs
pub struct S;

fn main() {}
