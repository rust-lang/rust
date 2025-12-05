#![crate_type = "rlib"]

pub fn inner<F>(f: F) -> F {
    (move || f)()
}
