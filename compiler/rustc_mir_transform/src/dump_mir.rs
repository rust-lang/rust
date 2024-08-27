//! This pass just dumps MIR at a specified point.

use std::fs::File;
use std::io;

use rustc_middle::mir::{write_mir_pretty, Body};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{OutFileName, OutputType};

pub(super) struct Marker(pub &'static str);

impl<'tcx> crate::MirPass<'tcx> for Marker {
    fn name(&self) -> &'static str {
        self.0
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {}
}

pub fn emit_mir(tcx: TyCtxt<'_>) -> io::Result<()> {
    match tcx.output_filenames(()).path(OutputType::Mir) {
        OutFileName::Stdout => {
            let mut f = io::stdout();
            write_mir_pretty(tcx, None, &mut f)?;
        }
        OutFileName::Real(path) => {
            let mut f = io::BufWriter::new(File::create(&path)?);
            write_mir_pretty(tcx, None, &mut f)?;
            if tcx.sess.opts.json_artifact_notifications {
                tcx.dcx().emit_artifact_notification(&path, "mir");
            }
        }
    }
    Ok(())
}
