
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Kind {
    /// Error if deallocated any other way than `rust_deallocate`
    Rust,
    /// Error if deallocated any other way than `free`
    C,
    /// Part of env var emulation
    Env,
}

impl Into<::rustc_miri::interpret::Kind<Kind>> for Kind {
    fn into(self) -> ::rustc_miri::interpret::Kind<Kind> {
        ::rustc_miri::interpret::Kind::Machine(self)
    }
}
