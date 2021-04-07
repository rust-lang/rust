use std::str::FromStr;

#[derive(Copy, Clone, Debug)]
pub enum CodegenMode {
    Aot,
    Jit,
    JitLazy,
}

impl Default for CodegenMode {
    fn default() -> Self {
        CodegenMode::Aot
    }
}

impl FromStr for CodegenMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "aot" => Ok(CodegenMode::Aot),
            "jit" => Ok(CodegenMode::Jit),
            "jit-lazy" => Ok(CodegenMode::JitLazy),
            _ => Err(format!("Unknown codegen mode `{}`", s)),
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BackendConfig {
    pub codegen_mode: CodegenMode,
}

impl BackendConfig {
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        let mut config = BackendConfig::default();
        for opt in opts {
            if let Some((name, value)) = opt.split_once('=') {
                match name {
                    "mode" => config.codegen_mode = value.parse()?,
                    _ => return Err(format!("Unknown option `{}`", name)),
                }
            } else {
                return Err(format!("Invalid option `{}`", opt));
            }
        }
        Ok(config)
    }
}
