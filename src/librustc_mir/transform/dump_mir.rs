// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass just dumps MIR at a specified point.

use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::io;

use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::mir::transform::{DefIdPass, MirCtxt, MirSource, PassHook};
use rustc::session::config::{OutputFilenames, OutputType};
use rustc::ty::TyCtxt;
use util as mir_util;

pub struct Marker(pub &'static str);

impl DefIdPass for Marker {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass<'a, 'tcx: 'a>(&self, mir_cx: &MirCtxt<'a, 'tcx>) -> Mir<'tcx> {
        mir_cx.steal_previous_mir()
    }
}

pub struct Disambiguator {
    is_after: bool
}

impl fmt::Display for Disambiguator {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let title = if self.is_after { "after" } else { "before" };
        write!(formatter, "{}", title)
    }
}

pub struct DumpMir;

impl PassHook for DumpMir {
    fn on_mir_pass<'a, 'tcx: 'a>(&self,
                                 mir_cx: &MirCtxt<'a, 'tcx>,
                                 mir: Option<(DefId, &Mir<'tcx>)>)
    {
        let tcx = mir_cx.tcx();
        let suite = mir_cx.suite();
        let pass_num = mir_cx.pass_num();
        let pass = tcx.mir_passes.pass(suite, pass_num);
        let name = &pass.name();
        let source = match mir {
            None => mir_cx.source(),
            Some((def_id, _)) => {
                let id = tcx.hir.as_local_node_id(def_id)
                                .expect("mir source requires local def-id");
                MirSource::from_node(tcx, id)
            }
        };
        if mir_util::dump_enabled(tcx, name, source) {
            let previous_mir;
            let mir_to_dump = match mir {
                Some((_, m)) => m,
                None => {
                    previous_mir = mir_cx.read_previous_mir();
                    &*previous_mir
                }
            };
            mir_util::dump_mir(tcx,
                               Some((suite, pass_num)),
                               name,
                               &Disambiguator { is_after: mir.is_some() },
                               source,
                               mir_to_dump);
        }
    }
}

pub fn emit_mir<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    outputs: &OutputFilenames)
    -> io::Result<()>
{
    let path = outputs.path(OutputType::Mir);
    let mut f = File::create(&path)?;
    mir_util::write_mir_pretty(tcx, None, &mut f)?;
    Ok(())
}
