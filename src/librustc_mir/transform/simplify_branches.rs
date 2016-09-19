// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that simplifies branches when their condition is known.

use rustc::ty::TyCtxt;
use rustc::middle::const_val::ConstVal;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::mir::*;

use std::fmt;

pub struct SimplifyBranches<'a> { label: &'a str }

impl<'a> SimplifyBranches<'a> {
    pub fn new(label: &'a str) -> Self {
        SimplifyBranches { label: label }
    }
}

impl<'l, 'tcx> MirPass<'tcx> for SimplifyBranches<'l> {
    fn run_pass<'a>(&mut self, _tcx: TyCtxt<'a, 'tcx, 'tcx>, _src: MirSource, mir: &mut Mir<'tcx>) {
        for block in mir.basic_blocks_mut() {
            let terminator = block.terminator_mut();
            terminator.kind = match terminator.kind {
                TerminatorKind::If { ref targets, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }) } => {
                    if cond {
                        TerminatorKind::Goto { target: targets.0 }
                    } else {
                        TerminatorKind::Goto { target: targets.1 }
                    }
                }

                TerminatorKind::Assert { target, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }), expected, .. } if cond == expected => {
                    TerminatorKind::Goto { target: target }
                }

                _ => continue
            };
        }
    }
}

impl<'l> Pass for SimplifyBranches<'l> {
    fn disambiguator<'a>(&'a self) -> Option<Box<fmt::Display+'a>> {
        Some(Box::new(self.label))
    }

    // avoid calling `type_name` - it contains `<'static>`
    fn name(&self) -> ::std::borrow::Cow<'static, str> { "SimplifyBranches".into() }
}
