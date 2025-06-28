//@ known-bug: #110395
// FIXME check-pass

#![feature(const_trait_impl, const_closures)]
#![allow(incomplete_features)]

const fn test() -> impl [const] Fn() {
    const move || {}
}

fn main() {}
