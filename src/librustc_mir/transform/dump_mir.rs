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

use std::fmt;

use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc::mir::transform::{Pass, MirPass, MirPassHook, MirSource};
use pretty;

pub struct Marker<'a>(pub &'a str);

impl<'b, 'tcx> MirPass<'tcx> for Marker<'b> {
    fn run_pass<'a>(&mut self, _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    _src: MirSource, _mir: &mut Mir<'tcx>)
    {}
}

impl<'b> Pass for Marker<'b> {
    fn name(&self) -> ::std::borrow::Cow<'static, str> { String::from(self.0).into() }
}

pub struct Disambiguator<'a> {
    pass: &'a Pass,
    is_after: bool
}

impl<'a> fmt::Display for Disambiguator<'a> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        let title = if self.is_after { "after" } else { "before" };
        if let Some(fmt) = self.pass.disambiguator() {
            write!(formatter, "{}-{}", fmt, title)
        } else {
            write!(formatter, "{}", title)
        }
    }
}

pub struct DumpMir;

impl<'tcx> MirPassHook<'tcx> for DumpMir {
    fn on_mir_pass<'a>(
        &mut self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        src: MirSource,
        mir: &Mir<'tcx>,
        pass: &Pass,
        is_after: bool)
    {
        pretty::dump_mir(
            tcx,
            &*pass.name(),
            &Disambiguator {
                pass: pass,
                is_after: is_after
            },
            src,
            mir
        );
    }
}

impl<'b> Pass for DumpMir {}
