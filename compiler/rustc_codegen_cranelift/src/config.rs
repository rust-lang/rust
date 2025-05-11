/// Configuration of cg_clif as passed in through `-Cllvm-args` and various env vars.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    /// Should the crate be AOT compiled or JIT executed.
    ///
    /// Defaults to AOT compilation. Can be set using `-Cllvm-args=jit-mode`.
    pub jit_mode: bool,

    /// When JIT mode is enable pass these arguments to the program.
    ///
    /// Defaults to the value of `CG_CLIF_JIT_ARGS`.
    pub jit_args: Vec<String>,
}

impl BackendConfig {
    /// Parse the configuration passed in using `-Cllvm-args`.
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        let mut config = BackendConfig {
            jit_mode: false,
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
            match &**opt {
                "jit-mode" => config.jit_mode = true,
                _ => return Err(format!("Unknown option `{}`", opt)),
            }
        }

        Ok(config)
    }
}
