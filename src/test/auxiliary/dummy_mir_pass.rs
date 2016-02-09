// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host

#![feature(plugin_registrar, rustc_private)]
#![feature(box_syntax)]

#[macro_use] extern crate rustc;
extern crate rustc_front;
extern crate rustc_plugin;
extern crate syntax;

use rustc::mir::transform::MirPass;
use rustc::mir::repr::{Mir, Literal};
use rustc::mir::visit::MutVisitor;
use rustc::middle::ty;
use rustc::middle::const_eval::ConstVal;
use rustc::lint::{LateContext, LintContext, LintPass, LateLintPass, LateLintPassObject, LintArray};
use rustc_plugin::Registry;
use rustc_front::hir;
use syntax::attr;

struct Pass;

impl MirPass for Pass {
    fn run_on_mir<'tcx>(&mut self, mir: &mut Mir<'tcx>, tcx: &ty::ctxt<'tcx>) {
        Visitor.visit_mir(mir)
    }
}

struct Visitor;

impl<'tcx> MutVisitor<'tcx> for Visitor {
    fn visit_literal(&mut self, literal: &mut Literal<'tcx>) {
        if let Literal::Value { value: ConstVal::Int(ref mut i @ 11) } = *literal {
            *i = 42;
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_mir_pass(box Pass);
}
