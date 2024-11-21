use std::str::FromStr;

/// The mode to use for compilation.
#[derive(Copy, Clone, Debug)]
pub enum CodegenMode {
    /// AOT compile the crate. This is the default.
    Aot,
    /// JIT compile and execute the crate.
    Jit,
    /// JIT compile and execute the crate, but only compile functions the first time they are used.
    JitLazy,
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

/// Configuration of cg_clif as passed in through `-Cllvm-args` and various env vars.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    /// Should the crate be AOT compiled or JIT executed.
    ///
    /// Defaults to AOT compilation. Can be set using `-Cllvm-args=mode=...`.
    pub codegen_mode: CodegenMode,

    /// When JIT mode is enable pass these arguments to the program.
    ///
    /// Defaults to the value of `CG_CLIF_JIT_ARGS`.
    pub jit_args: Vec<String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        BackendConfig {
            codegen_mode: CodegenMode::Aot,
            jit_args: {
                match std::env::var("CG_CLIF_JIT_ARGS") {
                    Ok(args) => args.split(' ').map(|arg| arg.to_string()).collect(),
                    Err(std::env::VarError::NotPresent) => vec![],
                    Err(std::env::VarError::NotUnicode(s)) => {
                        panic!("CG_CLIF_JIT_ARGS not unicode: {:?}", s);
                    }
                }
            },
        }
    }
}

impl BackendConfig {
    /// Parse the configuration passed in using `-Cllvm-args`.
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        let mut config = BackendConfig::default();
        for opt in opts {
            if opt.starts_with("-import-instr-limit") {
                // Silently ignore -import-instr-limit. It is set by rust's build system even when
                // testing cg_clif.
                continue;
            }
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
