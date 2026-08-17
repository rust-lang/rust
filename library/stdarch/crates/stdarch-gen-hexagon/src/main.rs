//! Hexagon code generator.
//!
//! Single binary that produces every generated file under
//! `core_arch/src/hexagon/`: scalar.rs (scalar intrinsics) and
//! v64.rs / v128.rs (HVX intrinsics).
//!
//! Run in check or bless mode via `STDARCH_GEN_MODE`.

mod hvx;
mod scalar;

use std::path::PathBuf;
use stdarch_gen_common::{run_generator, Mode};

fn main() -> Result<(), String> {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    let hexagon_dir = crate_dir.join("../core_arch/src/hexagon");
    // Either "check" to check the output versus the committed output, or "bless"
    // to update the output.
    let mode = Mode::from_env();

    run_generator(&hexagon_dir, mode, |out_dir| -> Result<(), String> {
        // Here scalar::generate writes scalar.rs .
        scalar::generate(&crate_dir, out_dir)?;
        // Here hvx::generate writes v64.rs and v128.rs .
        hvx::generate(&crate_dir, out_dir)?;
        Ok(())
    })
    .map_err(|e| e.to_string())?;

    Ok(())
}
