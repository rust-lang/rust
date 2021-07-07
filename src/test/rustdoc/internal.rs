// compile-flags: -Z force-unstable-if-unmarked

// Check that the unstable marker is not added for "rustc_private".

// @!matches internal/index.html \
//      '//*[@class="item-right docblock-short"]/span[@class="stab unstable"]'
// @!matches internal/index.html \
//      '//*[@class="item-right docblock-short"]/span[@class="stab internal"]'
// @matches - '//*[@class="item-right docblock-short"]' 'Docs'

// @!has internal/struct.S.html '//*[@class="stab unstable"]'
// @!has internal/struct.S.html '//*[@class="stab internal"]'
/// Docs
pub struct S;

fn main() {}
