pub(crate) enum PkgType {
    RustSrc,
    Cargo,
    Rls,
    RustAnalyzer,
    Clippy,
    Rustfmt,
    LlvmTools,
    Miri,
    Other(String),
}

impl PkgType {
    pub(crate) fn from_component(component: &str) -> Self {
        match component {
            "rust-src" => PkgType::RustSrc,
            "cargo" => PkgType::Cargo,
            "rls" | "rls-preview" => PkgType::Rls,
            "rust-analyzer" | "rust-analyzer-preview" => PkgType::RustAnalyzer,
            "clippy" | "clippy-preview" => PkgType::Clippy,
            "rustfmt" | "rustfmt-preview" => PkgType::Rustfmt,
            "llvm-tools" | "llvm-tools-preview" => PkgType::LlvmTools,
            "miri" | "miri-preview" => PkgType::Miri,
            other => PkgType::Other(other.into()),
        }
    }
}
