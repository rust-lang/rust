// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
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
#![deny(plugin_as_library)] // should have no effect in a plugin crate

extern crate rustc;
extern crate rustc_plugin;


use rustc::mir::*;
use rustc::ty::{Ty, TyCtxt, FnSig, subst::Substs, };
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
  let codegen = Box::new(GenericCountMismatch) as Box<_>;
  reg.register_intrinsic("generic_count_mismatch".into(), codegen);
  let codegen = Box::new(InputOutputMismatch) as Box<_>;
  reg.register_intrinsic("type_mismatch".into(), codegen);
}

struct GenericCountMismatch;
impl PluginIntrinsicCodegen for GenericCountMismatch {
  fn codegen_simple_intrinsic<'a, 'tcx>(&self,
                                        _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        _source_info: SourceInfo,
                                        _sig: &FnSig<'tcx>,
                                        _parent_mir: &Mir<'tcx>,
                                        _parent_param_substs: &'tcx Substs<'tcx>,
                                        _args: &Vec<Operand<'tcx>>,
                                        _dest: Place<'tcx>,
                                        _extra_stmts: &mut Vec<StatementKind<'tcx>>)
    where 'tcx: 'a,
  {
    unreachable!()
  }

  /// The number of generic parameters expected.
  fn generic_parameter_count<'a, 'tcx>(&self, _tcx: TyCtxt<'a, 'tcx, 'tcx>) -> usize { 5 }
  /// The types of the input args.
  fn inputs<'a, 'tcx>(&self, _tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<Ty<'tcx>> { vec![] }
  /// The return type.
  fn output<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
    tcx.mk_nil()
  }
}

struct InputOutputMismatch;
impl PluginIntrinsicCodegen for InputOutputMismatch {
  fn codegen_simple_intrinsic<'a, 'tcx>(&self,
                                        _tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        _source_info: SourceInfo,
                                        _sig: &FnSig<'tcx>,
                                        _parent_mir: &Mir<'tcx>,
                                        _parent_param_substs: &'tcx Substs<'tcx>,
                                        _args: &Vec<Operand<'tcx>>,
                                        _dest: Place<'tcx>,
                                        _extra_stmts: &mut Vec<StatementKind<'tcx>>)
    where 'tcx: 'a,
  {
    unreachable!()
  }

  /// The number of generic parameters expected.
  fn generic_parameter_count<'a, 'tcx>(&self, _tcx: TyCtxt<'a, 'tcx, 'tcx>) -> usize { 0 }
  /// The types of the input args.
  fn inputs<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<Ty<'tcx>> {
    vec![tcx.types.u64]
  }
  /// The return type.
  fn output<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
    tcx.types.u64
  }
}
