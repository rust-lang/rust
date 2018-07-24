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
use rustc::ty::{Ty, TyCtxt, FnSig, Const, subst::Substs, ParamEnv, };
use rustc_plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
  let codegen = Box::new(GetSecretValueCodegen) as Box<_>;
  reg.register_intrinsic("get_secret_value".into(), codegen);
}

struct GetSecretValueCodegen;
impl PluginIntrinsicCodegen for GetSecretValueCodegen {
  fn codegen_simple_intrinsic<'a, 'tcx>(&self,
                                        tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                        source_info: SourceInfo,
                                        _sig: &FnSig<'tcx>,
                                        _parent_mir: &Mir<'tcx>,
                                        _parent_param_substs: &'tcx Substs<'tcx>,
                                        _args: &Vec<Operand<'tcx>>,
                                        dest: Place<'tcx>,
                                        extra_stmts: &mut Vec<StatementKind<'tcx>>)
    where 'tcx: 'a,
  {
    // chosen by fair dice roll.
    // guaranteed to be random.
    const SECRET_VALUE: u64 = 4;

    let v = Const::from_bits(tcx, SECRET_VALUE as u128,
                             ParamEnv::empty().and(tcx.types.u64));
    let v = tcx.mk_const(*v);
    let v = Literal::Value {
      value: v,
    };
    let v = Constant {
      span: source_info.span,
      ty: tcx.types.u64,
      literal: v,
    };
    let v = Box::new(v);
    let v = Operand::Constant(v);
    let ret = Rvalue::Use(v);

    let stmt = StatementKind::Assign(dest, ret);
    extra_stmts.push(stmt);
  }

  /// The number of generic parameters expected.
  fn generic_parameter_count<'a, 'tcx>(&self, _tcx: TyCtxt<'a, 'tcx, 'tcx>) -> usize { 0 }
  /// The types of the input args.
  fn inputs<'a, 'tcx>(&self, _tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Vec<Ty<'tcx>> { vec![] }
  /// The return type.
  fn output<'a, 'tcx>(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
    tcx.types.u64
  }
}
