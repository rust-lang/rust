pub(crate) mod arg;
mod commands;

use std::path::Path;

use anyhow::Result;
use clap::Parser;
use path_macro::path;

use arg::Subcommands;

struct AutoConfig {
    toolchain: bool,
    fmt: bool,
    clippy: bool,
}

impl Subcommands {
    fn run_auto_things(&self) -> bool {
        use Subcommands::*;
        match self {
            // Early commands, that don't do auto-things and don't want the environment-altering things happening below.
            Toolchain { .. }
            | RustcPull { .. }
            | RustcPush { .. }
            | ManySeeds { .. }
            | Bench { .. } => false,
            Install { .. }
            | Check { .. }
            | Build { .. }
            | Test { .. }
            | Run { .. }
            | Fmt { .. }
            | Clippy { .. }
            | Cargo { .. } => true,
        }
    }
    fn get_config(&self, miri_dir: &Path) -> Option<AutoConfig> {
        let skip_auto_ops = std::env::var_os("MIRI_AUTO_OPS").is_some();
        if !self.run_auto_things() {
            return None;
        }
        if skip_auto_ops {
            return Some(AutoConfig { toolchain: false, fmt: false, clippy: false });
        }

        let auto_everything = path!(miri_dir / ".auto_everything").exists();
        let toolchain = auto_everything || path!(miri_dir / ".auto-toolchain").exists();
        let fmt = auto_everything || path!(miri_dir / ".auto-fmt").exists();
        let clippy = auto_everything || path!(miri_dir / ".auto-clippy").exists();
        Some(AutoConfig { toolchain, fmt, clippy })
    }
}
fn main() -> Result<()> {
    let args = arg::Cli::parse();
    commands::MiriRunner::exec(&args.commands)?;
    Ok(())
}
