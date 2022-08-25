// check-pass
// compile-flags: -C opt-level=3

#![crate_type = "lib"]

fn lambda<T: Default>() -> T {
    if true && let Some(bar) = transform() {
        bar
    } else {
        T::default()
    }
}

fn transform<T>() -> Option<T> {
    None
}
