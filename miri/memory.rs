
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum MemoryKind {
    /// Error if deallocated any other way than `rust_deallocate`
    Rust,
    /// Error if deallocated any other way than `free`
    C,
    /// Part of env var emulation
    Env,
}

impl Into<::rustc_miri::interpret::MemoryKind<MemoryKind>> for MemoryKind {
    fn into(self) -> ::rustc_miri::interpret::MemoryKind<MemoryKind> {
        ::rustc_miri::interpret::MemoryKind::Machine(self)
    }
}
