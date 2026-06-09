//@ check-pass
//@aux-build:once_cell.rs

#![warn(clippy::non_std_lazy_statics)]

// Should not error, since we used a type besides `sync::Lazy`
fn use_once_cell_race(x: once_cell::race::OnceBox<String>) {
    let _foo = x.get();
}

use once_cell::sync::Lazy;

static LAZY_BAZ: Lazy<String> = Lazy::new(|| "baz".to_uppercase());
