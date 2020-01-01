fn main() {
    // This is technically not sound -- but we're literally trying to test
    // that the sanitizer catches this, so I guess "intentionally unsound"?
    #[allow(deprecated)]
    let xs: [u8; 4] = unsafe { std::mem::uninitialized() };
    let y = xs[0] + xs[1];
}
