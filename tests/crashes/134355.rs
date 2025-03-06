//@ known-bug: #134355

//@compile-flags: --crate-type=lib
fn digit() -> str {
    return { i32::MIN };
}
