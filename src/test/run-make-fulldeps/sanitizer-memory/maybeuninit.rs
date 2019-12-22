use std::mem::MaybeUninit;

fn main() {
    // This is technically not sound -- but we're literally trying to test
    // that the sanitizer catches this, so I guess "intentionally unsound"?
    let xs: [u8; 4] = unsafe { MaybeUninit::uninit().assume_init() };
    let y = xs[0] + xs[1];
}
