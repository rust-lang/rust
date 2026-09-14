mod hvx;
mod scalar;

use clap::Parser;
use std::path::PathBuf;
use stdarch_gen_common::{run_generator, GeneratorCtx, Mode};

/// Hexagon code generator.
///
/// Produces every generated file under
/// `core_arch/src/hexagon/`: scalar.rs (scalar intrinsics) and
/// v64.rs / v128.rs (HVX intrinsics).
///
/// Run in check or bless mode via `STDARCH_GEN_MODE`.
#[derive(clap::Parser)]
struct Args {
    /// Generation mode.
    #[arg(long, env = "STDARCH_GEN_MODE")]
    mode: Option<Mode>,
    /// Path to a rustfmt binary that will be used to reformat the generated code.
    /// If unset, it will just use "rustfmt" from the environment.
    #[arg(long)]
    rustfmt_path: Option<PathBuf>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    let hexagon_dir = crate_dir.join("../core_arch/src/hexagon");
    // Either "check" to check the output versus the committed output, or "bless"
    // to update the output.
    let mode = args.mode.unwrap_or_default();
    let ctx = GeneratorCtx::new(args.rustfmt_path);

    run_generator(&ctx, &hexagon_dir, mode, |out_dir| -> Result<(), String> {
        // Here scalar::generate writes scalar.rs .
        scalar::generate(&crate_dir, out_dir)?;
        // Here hvx::generate writes v64.rs and v128.rs .
        hvx::generate(&crate_dir, out_dir)?;
        Ok(())
    })
    .map_err(|e| e.to_string())?;

    Ok(())
}
