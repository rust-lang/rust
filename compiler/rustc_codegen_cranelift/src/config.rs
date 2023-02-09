use std::env;
use std::str::FromStr;

fn bool_env_var(key: &str) -> bool {
    env::var(key).as_deref() == Ok("1")
}

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

    /// Enable the Cranelift ir verifier for all compilation passes. If not set it will only run
    /// once before passing the clif ir to Cranelift for compilation.
    ///
    /// Defaults to true when the `CG_CLIF_ENABLE_VERIFIER` env var is set to 1 or when cg_clif is
    /// compiled with debug assertions enabled or false otherwise. Can be set using
    /// `-Cllvm-args=enable_verifier=...`.
    pub enable_verifier: bool,

    /// Don't cache object files in the incremental cache. Useful during development of cg_clif
    /// to make it possible to use incremental mode for all analyses performed by rustc without
    /// caching object files when their content should have been changed by a change to cg_clif.
    ///
    /// Defaults to true when the `CG_CLIF_DISABLE_INCR_CACHE` env var is set to 1 or false
    /// otherwise. Can be set using `-Cllvm-args=disable_incr_cache=...`.
    pub disable_incr_cache: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        BackendConfig {
            codegen_mode: CodegenMode::Aot,
            jit_args: {
                let args = std::env::var("CG_CLIF_JIT_ARGS").unwrap_or_else(|_| String::new());
                args.split(' ').map(|arg| arg.to_string()).collect()
            },
            enable_verifier: cfg!(debug_assertions) || bool_env_var("CG_CLIF_ENABLE_VERIFIER"),
            disable_incr_cache: bool_env_var("CG_CLIF_DISABLE_INCR_CACHE"),
        }
    }
}

impl BackendConfig {
    /// Parse the configuration passed in using `-Cllvm-args`.
    pub fn from_opts(opts: &[String]) -> Result<Self, String> {
        fn parse_bool(name: &str, value: &str) -> Result<bool, String> {
            value.parse().map_err(|_| format!("failed to parse value `{}` for {}", value, name))
        }

        let mut config = BackendConfig::default();
        for opt in opts {
            if let Some((name, value)) = opt.split_once('=') {
                match name {
                    "mode" => config.codegen_mode = value.parse()?,
                    "enable_verifier" => config.enable_verifier = parse_bool(name, value)?,
                    "disable_incr_cache" => config.disable_incr_cache = parse_bool(name, value)?,
                    _ => return Err(format!("Unknown option `{}`", name)),
                }
            } else {
                return Err(format!("Invalid option `{}`", opt));
            }
        }

        Ok(config)
    }
}
