// build-pass (FIXME(62277): could be check-pass?)

// tests that the following code compiles, but produces a future-compatibility warning

fn main() {
    let data = std::ptr::null();
    let _ = &data as *const *const ();
    if data.is_null() {}
}
