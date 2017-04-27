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

use rustc::hir::def_id::{DefId, LOCAL_CRATE};
use rustc::session::config::{OutputFilenames, OutputType};
use rustc::ty::TyCtxt;
use rustc::mir::transform::{DefIdPass, Pass, PassHook, MirSource};
use util as mir_util;

pub struct Marker(pub &'static str);

impl DefIdPass for Marker {
    fn name<'a>(&'a self) -> Cow<'a, str> {
        Cow::Borrowed(self.0)
    }

    fn run_pass<'a, 'tcx>(&self, _: TyCtxt<'a, 'tcx, 'tcx>, _: DefId) {
        // no-op
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
    fn on_mir_pass<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        pass: &Pass,
        pass_num: usize,
        is_after: bool)
    {
        // No dump filters enabled.
        if tcx.sess.opts.debugging_opts.dump_mir.is_none() {
            return;
        }

        for &def_id in tcx.mir_keys(LOCAL_CRATE).iter() {
            let id = tcx.hir.as_local_node_id(def_id).unwrap();
            let source = MirSource::from_node(tcx, id);
            let mir = tcx.item_mir(def_id);
            mir_util::dump_mir(
                tcx,
                pass_num,
                &*pass.name(),
                &Disambiguator { is_after },
                source,
                &mir
            );
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
