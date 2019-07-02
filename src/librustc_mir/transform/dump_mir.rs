//! This pass just dumps MIR at a specified point.

use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::io;

use rustc::mir::Body;
use rustc::session::config::{OutputFilenames, OutputType};
use rustc::ty::TyCtxt;
use crate::transform::{MirPass, MirSource};
use crate::util as mir_util;

pub struct Marker(pub &'static str);

impl MirPass for Marker {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass<'tcx>(&self, _tcx: TyCtxt<'tcx>, _source: MirSource<'tcx>, _body: &mut Body<'tcx>) {
    }
}

pub struct Disambiguator {
    is_after: bool
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
    source: MirSource<'tcx>,
    body: &Body<'tcx>,
    is_after: bool,
) {
    if mir_util::dump_enabled(tcx, pass_name, source) {
        mir_util::dump_mir(tcx,
                           Some(pass_num),
                           pass_name,
                           &Disambiguator { is_after },
                           source,
                           body,
                           |_, _| Ok(()) );
    }
}

pub fn emit_mir(tcx: TyCtxt<'_>, outputs: &OutputFilenames) -> io::Result<()> {
    let path = outputs.path(OutputType::Mir);
    let mut f = File::create(&path)?;
    mir_util::write_mir_pretty(tcx, None, &mut f)?;
    Ok(())
}
