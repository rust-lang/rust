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

impl BackendConfig {
    /// Parse the configuration passed in using `-Cllvm-args`.
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        let mut config = BackendConfig {
            codegen_mode: CodegenMode::Aot,
            jit_args: match std::env::var("CG_CLIF_JIT_ARGS") {
                Ok(args) => args.split(' ').map(|arg| arg.to_string()).collect(),
                Err(std::env::VarError::NotPresent) => vec![],
                Err(std::env::VarError::NotUnicode(s)) => {
                    panic!("CG_CLIF_JIT_ARGS not unicode: {:?}", s);
                }
            },
        };

        for opt in opts {
            if opt.starts_with("-import-instr-limit") {
                // Silently ignore -import-instr-limit. It is set by rust's build system even when
                // testing cg_clif.
                continue;
            }
            if let Some((name, value)) = opt.split_once('=') {
                match name {
                    "mode" => {
                        config.codegen_mode = match value {
                            "aot" => CodegenMode::Aot,
                            "jit" => CodegenMode::Jit,
                            "jit-lazy" => CodegenMode::JitLazy,
                            _ => return Err(format!("Unknown codegen mode `{}`", value)),
                        };
                    }
                    _ => return Err(format!("Unknown option `{}`", name)),
                }
            } else {
                return Err(format!("Invalid option `{}`", opt));
            }
        }

        Ok(config)
    }
}
