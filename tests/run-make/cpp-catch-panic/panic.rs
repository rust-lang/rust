unsafe extern "C-unwind" {
    safe fn caught_panic() -> bool;
}

#[unsafe(no_mangle)]
extern "C-unwind" fn rust_panic() {
    panic!();
}

fn main() {
    assert!(caught_panic());
}
