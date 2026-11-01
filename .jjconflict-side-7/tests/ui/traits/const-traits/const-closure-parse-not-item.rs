//@check-pass

#![feature(const_trait_impl, const_closures)]

const fn test() -> impl [const] Fn() {
    const move || {}
}

fn main() {}
