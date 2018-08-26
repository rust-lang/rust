#[derive(Debug, PartialEq, Copy, Clone, Hash, Eq)]
pub enum MemoryKind {
    /// `__rust_alloc` memory
    Rust,
    /// `malloc` memory
    C,
    /// Part of env var emulation
    Env,
    // mutable statics
    MutStatic,
}

impl Into<::rustc_mir::interpret::MemoryKind<MemoryKind>> for MemoryKind {
    fn into(self) -> ::rustc_mir::interpret::MemoryKind<MemoryKind> {
        ::rustc_mir::interpret::MemoryKind::Machine(self)
    }
}
