// `exit` with `()` return type should not ICE.
// Regression test for <https://github.com/rust-lang/miri/issues/5563>.

#[allow(suspicious_runtime_symbol_definitions)]
unsafe extern "C" {
    fn exit(code: i32) -> ();
}

fn main() {
    unsafe {
        exit(0);
    }
}
