// The sheer existence of this file is an awful hack. See the comments in
// `src/bootstrap/native.rs` for why this is needed when compiling LLD.

use std::env;
use std::io::{self, Write};
use std::process::{self, Command, Stdio};

fn main() {
    let real_llvm_config = env::var_os("LLVM_CONFIG_REAL").unwrap();
    let mut cmd = Command::new(real_llvm_config);
    cmd.args(env::args().skip(1)).stderr(Stdio::piped());
    let output = cmd.output().expect("failed to spawn llvm-config");
    let mut stdout = String::from_utf8_lossy(&output.stdout);

    if let Ok(to_replace) = env::var("LLVM_CONFIG_SHIM_REPLACE") {
        if let Ok(replace_with) = env::var("LLVM_CONFIG_SHIM_REPLACE_WITH") {
            stdout = stdout.replace(&to_replace, &replace_with).into();
        }
    }

    print!("{}", stdout.replace("\\", "/"));
    io::stdout().flush().unwrap();
    process::exit(output.status.code().unwrap_or(1));
}
