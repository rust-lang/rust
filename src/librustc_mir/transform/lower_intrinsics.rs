// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some intrinsics can be lowered to MIR code. Doing this enables
//! more optimizations to happen.

use std::borrow::Cow;

use rustc::middle::const_val::ConstVal;
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::ty::{TyCtxt, TypeVariants};
use syntax::abi::Abi;
use syntax_pos::symbol::InternedString;

pub struct LowerIntrinsics;

impl Pass for LowerIntrinsics {
    fn name(&self) -> Cow<'static, str> { "LowerIntrinsics".into() }
}

impl<'tcx> MirPass<'tcx> for LowerIntrinsics {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, _: MirSource, mir: &mut Mir<'tcx>) {
        for bb_data in mir.basic_blocks_mut() {
            if let Some((func, args, target)) = intrinsic_call(bb_data) {
                lower_intrinsic(tcx, bb_data, func, args, target);
            }
        }
    }
}

// Returns the elements of the function call in case it is an intrinsic
fn intrinsic_call<'tcx>(bb_data: &BasicBlockData<'tcx>)
                        -> Option<(Constant<'tcx>, Vec<Operand<'tcx>>, BasicBlock)> {
    use self::TerminatorKind::Call;
    match bb_data.terminator().kind {
        Call { func: Operand::Constant(ref func), ref args, ref destination, .. } => {
            // Note: for all intrinsics that we lower in this pass, we assume that they
            // don't require cleanup. If an intrinsic happens to require cleanup,
            // the cleanup code will be happily ignored and bad things will happen.

            match func.ty.sty {
                TypeVariants::TyFnDef(_, _, fn_sig)
                if fn_sig.skip_binder().abi == Abi::RustIntrinsic => {
                    let target = match destination {
                        &Some(ref d) => d.1,
                        &None => {
                            // Ignore diverging intrinsics
                            return None
                        }
                    };

                    Some((func.clone(), args.clone(), target))
                }
                _ => None
            }
        }
        _ => None
    }
}

// Returns the name of the intrinsic that is being called
fn intrinsic_name<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, func: &Constant<'tcx>) -> InternedString {
    let def_id = match func.literal {
        Literal::Item { def_id, .. } => def_id,
        Literal::Promoted { .. } => span_bug!(func.span, "promoted value with rust intrinsic abi"),
        Literal::Value { ref value } => match value {
            &ConstVal::Function(def_id, _) => def_id,
            _ => span_bug!(func.span, "literal value with rust intrinsic abi"),
        }
    };

    tcx.item_name(def_id).as_str()
}

fn lower_intrinsic<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             basic_block: &mut BasicBlockData<'tcx>,
                             func: Constant<'tcx>,
                             args: Vec<Operand<'tcx>>,
                             target: BasicBlock) {
    let name = &*intrinsic_name(tcx, &func);
    if name == "move_val_init" {
        assert_eq!(args.len(), 2);

        if let Operand::Consume(dest) = args[0].clone() {
            // move_val_init(dest, src)
            //   =>
            // Assign(dest, src)
            let source_info = basic_block.terminator().source_info;
            basic_block.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(
                    Lvalue::Projection(Box::new(Projection {
                        base: dest,
                        elem: ProjectionElem::Deref
                    })),
                    Rvalue::Use(args[1].clone())
                )
            });
        } else {
            bug!("destination argument not lvalue?");
        }

        // Since this is no longer a function call, replace the
        // terminator with a simple Goto
        basic_block.terminator_mut().kind = TerminatorKind::Goto { target };
    }
}
