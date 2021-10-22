//! This pass just dumps MIR at a specified point.

use std::borrow::Cow;
use std::fs::File;
use std::io;

use crate::MirPass;
use rustc_middle::mir::write_mir_pretty;
use rustc_middle::mir::Body;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{OutputFilenames, OutputType};

pub struct Marker(pub &'static str);

impl<'tcx> MirPass<'tcx> for Marker {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {}
}

pub fn emit_mir(tcx: TyCtxt<'_>, outputs: &OutputFilenames) -> io::Result<()> {
    let path = outputs.path(OutputType::Mir);
    let mut f = io::BufWriter::new(File::create(&path)?);
    write_mir_pretty(tcx, None, &mut f)?;
    Ok(())
}
