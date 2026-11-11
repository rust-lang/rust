use std::str::FromStr;

/// Represents a codegen backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum CodegenBackendKind {
    #[default]
    Llvm,
    Cranelift,
    Gcc,
    Custom(String),
}

impl CodegenBackendKind {
    /// Name of the codegen backend, as identified in the `compiler` directory
    /// (`rustc_codegen_<name>`).
    pub(crate) fn name(&self) -> &str {
        match self {
            CodegenBackendKind::Llvm => "llvm",
            CodegenBackendKind::Cranelift => "cranelift",
            CodegenBackendKind::Gcc => "gcc",
            CodegenBackendKind::Custom(name) => name,
        }
    }

    /// Name of the codegen backend's crate, e.g. `rustc_codegen_cranelift`.
    pub(crate) fn crate_name(&self) -> String {
        format!("rustc_codegen_{}", self.name())
    }

    pub(crate) fn is_llvm(&self) -> bool {
        matches!(self, Self::Llvm)
    }
}

/// FIXME(Zalathar): This is partly redundant with the parsing code in `parse_codegen_backends`.
impl FromStr for CodegenBackendKind {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "" => Err("Invalid empty backend name"),
            "gcc" => Ok(Self::Gcc),
            "llvm" => Ok(Self::Llvm),
            "cranelift" => Ok(Self::Cranelift),
            _ => Ok(Self::Custom(s.to_string())),
        }
    }
}
