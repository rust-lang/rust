//! This pass just dumps MIR at a specified point.

use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::io;

use crate::MirPass;
use rustc_middle::mir::Body;
use rustc_middle::mir::{dump_enabled, dump_mir, write_mir_pretty};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::{OutputFilenames, OutputType};

pub struct Marker(pub &'static str);

impl<'tcx> MirPass<'tcx> for Marker {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass(&self, _tcx: TyCtxt<'tcx>, _body: &mut Body<'tcx>) {}
}

pub struct Disambiguator {
    is_after: bool,
}

impl fmt::Display for Disambiguator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title = if self.is_after { "after" } else { "before" };
        write!(formatter, "{}", title)
    }
}

pub fn on_mir_pass<'tcx>(
    tcx: TyCtxt<'tcx>,
    pass_num: &dyn fmt::Display,
    pass_name: &str,
    body: &Body<'tcx>,
    is_after: bool,
) {
    if dump_enabled(tcx, pass_name, body.source.def_id()) {
        dump_mir(tcx, Some(pass_num), pass_name, &Disambiguator { is_after }, body, |_, _| Ok(()));
    }
}

pub fn emit_mir(tcx: TyCtxt<'_>, outputs: &OutputFilenames) -> io::Result<()> {
    let path = outputs.path(OutputType::Mir);
    let mut f = io::BufWriter::new(File::create(&path)?);
    write_mir_pretty(tcx, None, &mut f)?;
    Ok(())
}
