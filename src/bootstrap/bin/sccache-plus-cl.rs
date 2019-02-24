use std::env;
use std::process::{self, Command};

fn main() {
    let target = env::var("SCCACHE_TARGET").unwrap();
    // Locate the actual compiler that we're invoking
    env::set_var("CC", env::var_os("SCCACHE_CC").unwrap());
    env::set_var("CXX", env::var_os("SCCACHE_CXX").unwrap());
    let mut cfg = cc::Build::new();
    cfg.cargo_metadata(false)
       .out_dir("/")
       .target(&target)
       .host(&target)
       .opt_level(0)
       .warnings(false)
       .debug(false);
    let compiler = cfg.get_compiler();

    // Invoke sccache with said compiler
    let sccache_path = env::var_os("SCCACHE_PATH").unwrap();
    let mut cmd = Command::new(&sccache_path);
    cmd.arg(compiler.path());
    for &(ref k, ref v) in compiler.env() {
        cmd.env(k, v);
    }
    for arg in env::args().skip(1) {
        cmd.arg(arg);
    }

    if let Ok(s) = env::var("SCCACHE_EXTRA_ARGS") {
        for s in s.split_whitespace() {
            cmd.arg(s);
        }
    }

    let status = cmd.status().expect("failed to spawn");
    process::exit(status.code().unwrap_or(2))
}
