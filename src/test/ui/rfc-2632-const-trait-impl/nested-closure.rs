// check-pass

#![feature(const_trait_impl, once_cell)]

use std::sync::LazyLock;

static EXTERN_FLAGS: LazyLock<String> = LazyLock::new(|| {
    let x = || String::new();
    x()
});

fn main() {}
