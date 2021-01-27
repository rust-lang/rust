//! Argument passing

use crate::prelude::*;
use crate::value_and_place::assert_assignable;

use cranelift_codegen::ir::ArgumentPurpose;
use rustc_target::abi::call::{ArgAbi, PassMode};
use smallvec::{smallvec, SmallVec};

pub(super) trait ArgAbiExt<'tcx> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> SmallVec<[AbiParam; 2]>;
    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>);
}

impl<'tcx> ArgAbiExt<'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> SmallVec<[AbiParam; 2]> {
        match self.mode {
            PassMode::Ignore => smallvec![],
            PassMode::Direct(_) => match &self.layout.abi {
                Abi::Scalar(scalar) => {
                    smallvec![AbiParam::new(scalar_to_clif_type(tcx, scalar.clone()))]
                }
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    smallvec![AbiParam::new(vector_ty)]
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Pair(_, _) => match &self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a.clone());
                    let b = scalar_to_clif_type(tcx, b.clone());
                    smallvec![AbiParam::new(a), AbiParam::new(b)]
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Cast(_) => smallvec![AbiParam::new(pointer_ty(tcx))],
            PassMode::Indirect {
                attrs: _,
                extra_attrs: None,
                on_stack,
            } => {
                if on_stack {
                    let size = u32::try_from(self.layout.size.bytes()).unwrap();
                    smallvec![AbiParam::special(
                        pointer_ty(tcx),
                        ArgumentPurpose::StructArgument(size),
                    )]
                } else {
                    smallvec![AbiParam::new(pointer_ty(tcx))]
                }
            }
            PassMode::Indirect {
                attrs: _,
                extra_attrs: Some(_),
                on_stack,
            } => {
                assert!(!on_stack);
                smallvec![
                    AbiParam::new(pointer_ty(tcx)),
                    AbiParam::new(pointer_ty(tcx)),
                ]
            }
        }
    }

    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>) {
        match self.mode {
            PassMode::Ignore => (None, vec![]),
            PassMode::Direct(_) => match &self.layout.abi {
                Abi::Scalar(scalar) => (
                    None,
                    vec![AbiParam::new(scalar_to_clif_type(tcx, scalar.clone()))],
                ),
                // FIXME implement Vector Abi in a cg_llvm compatible way
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    (None, vec![AbiParam::new(vector_ty)])
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Pair(_, _) => match &self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a.clone());
                    let b = scalar_to_clif_type(tcx, b.clone());
                    (None, vec![AbiParam::new(a), AbiParam::new(b)])
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Cast(_) => (
                Some(AbiParam::special(
                    pointer_ty(tcx),
                    ArgumentPurpose::StructReturn,
                )),
                vec![],
            ),
            PassMode::Indirect {
                attrs: _,
                extra_attrs: None,
                on_stack,
            } => {
                assert!(!on_stack);
                (
                    Some(AbiParam::special(
                        pointer_ty(tcx),
                        ArgumentPurpose::StructReturn,
                    )),
                    vec![],
                )
            }
            PassMode::Indirect {
                attrs: _,
                extra_attrs: Some(_),
                on_stack: _,
            } => unreachable!("unsized return value"),
        }
    }
}

/// Get a set of values to be passed as function arguments.
pub(super) fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    arg: CValue<'tcx>,
    arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
) -> SmallVec<[Value; 2]> {
    assert_assignable(fx, arg.layout().ty, arg_abi.layout.ty);
    match arg_abi.mode {
        PassMode::Ignore => smallvec![],
        PassMode::Direct(_) => smallvec![arg.load_scalar(fx)],
        PassMode::Pair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            smallvec![a, b]
        }
        PassMode::Cast(_) | PassMode::Indirect { .. } => match arg.force_stack(fx) {
            (ptr, None) => smallvec![ptr.get_addr(fx)],
            (ptr, Some(meta)) => smallvec![ptr.get_addr(fx), meta],
        },
    }
}

/// Create a [`CValue`] containing the value of a function parameter adding clif function parameters
/// as necessary.
pub(super) fn cvalue_for_param<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    start_block: Block,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] local: Option<mir::Local>,
    #[cfg_attr(not(debug_assertions), allow(unused_variables))] local_field: Option<usize>,
    arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
) -> Option<CValue<'tcx>> {
    let clif_types = arg_abi.get_abi_param(fx.tcx);
    let block_params = clif_types
        .into_iter()
        .map(|abi_param| fx.bcx.append_block_param(start_block, abi_param.value_type))
        .collect::<SmallVec<[_; 2]>>();

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        &block_params,
        arg_abi.mode,
        arg_abi.layout,
    );

    match arg_abi.mode {
        PassMode::Ignore => None,
        PassMode::Direct(_) => {
            assert_eq!(block_params.len(), 1, "{:?}", block_params);
            Some(CValue::by_val(block_params[0], arg_abi.layout))
        }
        PassMode::Pair(_, _) => {
            assert_eq!(block_params.len(), 2, "{:?}", block_params);
            Some(CValue::by_val_pair(
                block_params[0],
                block_params[1],
                arg_abi.layout,
            ))
        }
        PassMode::Cast(_)
        | PassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => {
            assert_eq!(block_params.len(), 1, "{:?}", block_params);
            Some(CValue::by_ref(
                Pointer::new(block_params[0]),
                arg_abi.layout,
            ))
        }
        PassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => {
            assert_eq!(block_params.len(), 2, "{:?}", block_params);
            Some(CValue::by_ref_unsized(
                Pointer::new(block_params[0]),
                block_params[1],
                arg_abi.layout,
            ))
        }
    }
}
