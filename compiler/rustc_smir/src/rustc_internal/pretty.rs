use std::io;

use rustc_middle::ty::TyCtxt;

use super::run;
use crate::stable_mir;

pub fn write_smir_pretty<'tcx, W: io::Write>(tcx: TyCtxt<'tcx>, w: &mut W) -> io::Result<()> {
    writeln!(
        w,
        "// WARNING: This is highly experimental output it's intended for stable-mir developers only."
    )?;
    writeln!(
        w,
        "// If you find a bug or want to improve the output open a issue at https://github.com/rust-lang/project-stable-mir."
    )?;
    let _ = run(tcx, || {
        let items = stable_mir::all_local_items();
        let _ = items.iter().map(|item| -> io::Result<()> { item.emit_mir(w) }).collect::<Vec<_>>();
    });
    Ok(())
}
