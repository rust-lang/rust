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
use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::io;

use rustc::session::config::{OutputFilenames, OutputType};
use rustc::ty::TyCtxt;
use rustc::mir::Mir;
use rustc::mir::transform::{DefIdPass, PassHook, MirCtxt};
use util as mir_util;

pub struct Marker(pub &'static str);

impl DefIdPass for Marker {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass<'a, 'tcx: 'a>(&self, mir_cx: &MirCtxt<'a, 'tcx>) -> &'tcx RefCell<Mir<'tcx>> {
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
                             mir: Option<&Mir<'tcx>>)
    {
        let tcx = mir_cx.tcx();
        let pass_set = mir_cx.pass_set();
        let pass_num = mir_cx.pass_num();
        let pass = tcx.mir_passes.pass(pass_set, pass_num);
        let name = &pass.name();
        let source = mir_cx.source();
        if mir_util::dump_enabled(tcx, name, source) {
            let previous_mir;
            let mir_to_dump = match mir {
                Some(m) => m,
                None => {
                    previous_mir = mir_cx.read_previous_mir();
                    &*previous_mir
                }
            };
            mir_util::dump_mir(tcx,
                               Some((pass_set, pass_num)),
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
