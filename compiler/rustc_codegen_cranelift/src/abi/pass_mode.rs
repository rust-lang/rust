//! Argument passing

use crate::prelude::*;
use crate::value_and_place::assert_assignable;

use cranelift_codegen::ir::{ArgumentExtension, ArgumentPurpose};
use rustc_target::abi::call::{
    ArgAbi, ArgAttributes, ArgExtension as RustcArgExtension, CastTarget, PassMode, Reg, RegKind,
};
use smallvec::{smallvec, SmallVec};

pub(super) trait ArgAbiExt<'tcx> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> SmallVec<[AbiParam; 2]>;
    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>);
}

fn reg_to_abi_param(reg: Reg) -> AbiParam {
    let clif_ty = match (reg.kind, reg.size.bytes()) {
        (RegKind::Integer, 1) => types::I8,
        (RegKind::Integer, 2) => types::I16,
        (RegKind::Integer, 3..=4) => types::I32,
        (RegKind::Integer, 5..=8) => types::I64,
        (RegKind::Integer, 9..=16) => types::I128,
        (RegKind::Float, 4) => types::F32,
        (RegKind::Float, 8) => types::F64,
        (RegKind::Vector, size) => types::I8.by(u32::try_from(size).unwrap()).unwrap(),
        _ => unreachable!("{:?}", reg),
    };
    AbiParam::new(clif_ty)
}

fn apply_arg_attrs_to_abi_param(mut param: AbiParam, arg_attrs: ArgAttributes) -> AbiParam {
    match arg_attrs.arg_ext {
        RustcArgExtension::None => {}
        RustcArgExtension::Zext => param.extension = ArgumentExtension::Uext,
        RustcArgExtension::Sext => param.extension = ArgumentExtension::Sext,
    }
    param
}

fn cast_target_to_abi_params(cast: &CastTarget) -> SmallVec<[AbiParam; 2]> {
    let (rest_count, rem_bytes) = if cast.rest.unit.size.bytes() == 0 {
        (0, 0)
    } else {
        (
            cast.rest.total.bytes() / cast.rest.unit.size.bytes(),
            cast.rest.total.bytes() % cast.rest.unit.size.bytes(),
        )
    };

    // Note: Unlike the LLVM equivalent of this code we don't have separate branches for when there
    // is no prefix as a single unit, an array and a heterogeneous struct are not represented using
    // different types in Cranelift IR. Instead a single array of primitive types is used.

    // Create list of fields in the main structure
    let mut args = cast
        .prefix
        .iter()
        .flatten()
        .map(|&reg| reg_to_abi_param(reg))
        .chain((0..rest_count).map(|_| reg_to_abi_param(cast.rest.unit)))
        .collect::<SmallVec<_>>();

    // Append final integer
    if rem_bytes != 0 {
        // Only integers can be really split further.
        assert_eq!(cast.rest.unit.kind, RegKind::Integer);
        args.push(reg_to_abi_param(Reg {
            kind: RegKind::Integer,
            size: Size::from_bytes(rem_bytes),
        }));
    }

    args
}

impl<'tcx> ArgAbiExt<'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> SmallVec<[AbiParam; 2]> {
        match self.mode {
            PassMode::Ignore => smallvec![],
            PassMode::Direct(attrs) => match self.layout.abi {
                Abi::Scalar(scalar) => smallvec![apply_arg_attrs_to_abi_param(
                    AbiParam::new(scalar_to_clif_type(tcx, scalar)),
                    attrs
                )],
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    smallvec![AbiParam::new(vector_ty)]
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Pair(attrs_a, attrs_b) => match self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a);
                    let b = scalar_to_clif_type(tcx, b);
                    smallvec![
                        apply_arg_attrs_to_abi_param(AbiParam::new(a), attrs_a),
                        apply_arg_attrs_to_abi_param(AbiParam::new(b), attrs_b),
                    ]
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Cast(ref cast, pad_i32) => {
                assert!(!pad_i32, "padding support not yet implemented");
                cast_target_to_abi_params(cast)
            }
            PassMode::Indirect { attrs, extra_attrs: None, on_stack } => {
                if on_stack {
                    // Abi requires aligning struct size to pointer size
                    let size = self.layout.size.align_to(tcx.data_layout.pointer_align.abi);
                    let size = u32::try_from(size.bytes()).unwrap();
                    smallvec![apply_arg_attrs_to_abi_param(
                        AbiParam::special(pointer_ty(tcx), ArgumentPurpose::StructArgument(size),),
                        attrs
                    )]
                } else {
                    smallvec![apply_arg_attrs_to_abi_param(AbiParam::new(pointer_ty(tcx)), attrs)]
                }
            }
            PassMode::Indirect { attrs, extra_attrs: Some(extra_attrs), on_stack } => {
                assert!(!on_stack);
                smallvec![
                    apply_arg_attrs_to_abi_param(AbiParam::new(pointer_ty(tcx)), attrs),
                    apply_arg_attrs_to_abi_param(AbiParam::new(pointer_ty(tcx)), extra_attrs),
                ]
            }
        }
    }

    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>) {
        match self.mode {
            PassMode::Ignore => (None, vec![]),
            PassMode::Direct(_) => match self.layout.abi {
                Abi::Scalar(scalar) => {
                    (None, vec![AbiParam::new(scalar_to_clif_type(tcx, scalar))])
                }
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    (None, vec![AbiParam::new(vector_ty)])
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Pair(_, _) => match self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a);
                    let b = scalar_to_clif_type(tcx, b);
                    (None, vec![AbiParam::new(a), AbiParam::new(b)])
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            PassMode::Cast(ref cast, _) => {
                (None, cast_target_to_abi_params(cast).into_iter().collect())
            }
            PassMode::Indirect { attrs: _, extra_attrs: None, on_stack } => {
                assert!(!on_stack);
                (Some(AbiParam::special(pointer_ty(tcx), ArgumentPurpose::StructReturn)), vec![])
            }
            PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
                unreachable!("unsized return value")
            }
        }
    }
}

pub(super) fn to_casted_value<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    arg: CValue<'tcx>,
    cast: &CastTarget,
) -> SmallVec<[Value; 2]> {
    let (ptr, meta) = arg.force_stack(fx);
    assert!(meta.is_none());
    let mut offset = 0;
    cast_target_to_abi_params(cast)
        .into_iter()
        .map(|param| {
            let val = ptr.offset_i64(fx, offset).load(fx, param.value_type, MemFlags::new());
            offset += i64::from(param.value_type.bytes());
            val
        })
        .collect()
}

pub(super) fn from_casted_value<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    block_params: &[Value],
    layout: TyAndLayout<'tcx>,
    cast: &CastTarget,
) -> CValue<'tcx> {
    let abi_params = cast_target_to_abi_params(cast);
    let abi_param_size: u32 = abi_params.iter().map(|param| param.value_type.bytes()).sum();
    let layout_size = u32::try_from(layout.size.bytes()).unwrap();
    let stack_slot = fx.bcx.create_sized_stack_slot(StackSlotData {
        kind: StackSlotKind::ExplicitSlot,
        // FIXME Don't force the size to a multiple of 16 bytes once Cranelift gets a way to
        // specify stack slot alignment.
        // Stack slot size may be bigger for for example `[u8; 3]` which is packed into an `i32`.
        // It may also be smaller for example when the type is a wrapper around an integer with a
        // larger alignment than the integer.
        size: (std::cmp::max(abi_param_size, layout_size) + 15) / 16 * 16,
    });
    let ptr = Pointer::stack_slot(stack_slot);
    let mut offset = 0;
    let mut block_params_iter = block_params.iter().copied();
    for param in abi_params {
        let val = ptr.offset_i64(fx, offset).store(
            fx,
            block_params_iter.next().unwrap(),
            MemFlags::new(),
        );
        offset += i64::from(param.value_type.bytes());
        val
    }
    assert_eq!(block_params_iter.next(), None, "Leftover block param");
    CValue::by_ref(ptr, layout)
}

/// Get a set of values to be passed as function arguments.
pub(super) fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    arg: CValue<'tcx>,
    arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    is_owned: bool,
) -> SmallVec<[Value; 2]> {
    assert_assignable(fx, arg.layout().ty, arg_abi.layout.ty, 16);
    match arg_abi.mode {
        PassMode::Ignore => smallvec![],
        PassMode::Direct(_) => smallvec![arg.load_scalar(fx)],
        PassMode::Pair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            smallvec![a, b]
        }
        PassMode::Cast(ref cast, _) => to_casted_value(fx, arg, cast),
        PassMode::Indirect { .. } => {
            if is_owned {
                match arg.force_stack(fx) {
                    (ptr, None) => smallvec![ptr.get_addr(fx)],
                    (ptr, Some(meta)) => smallvec![ptr.get_addr(fx), meta],
                }
            } else {
                // Ownership of the value at the backing storage for an argument is passed to the
                // callee per the ABI, so we must make a copy of the argument unless the argument
                // local is moved.
                let place = CPlace::new_stack_slot(fx, arg.layout());
                place.write_cvalue(fx, arg);
                smallvec![place.to_ptr().get_addr(fx)]
            }
        }
    }
}

/// Create a [`CValue`] containing the value of a function parameter adding clif function parameters
/// as necessary.
pub(super) fn cvalue_for_param<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    local: Option<mir::Local>,
    local_field: Option<usize>,
    arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
    block_params_iter: &mut impl Iterator<Item = Value>,
) -> Option<CValue<'tcx>> {
    let block_params = arg_abi
        .get_abi_param(fx.tcx)
        .into_iter()
        .map(|abi_param| {
            let block_param = block_params_iter.next().unwrap();
            assert_eq!(fx.bcx.func.dfg.value_type(block_param), abi_param.value_type);
            block_param
        })
        .collect::<SmallVec<[_; 2]>>();

    crate::abi::comments::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        &block_params,
        &arg_abi.mode,
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
            Some(CValue::by_val_pair(block_params[0], block_params[1], arg_abi.layout))
        }
        PassMode::Cast(ref cast, _) => {
            Some(from_casted_value(fx, &block_params, arg_abi.layout, cast))
        }
        PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ } => {
            assert_eq!(block_params.len(), 1, "{:?}", block_params);
            Some(CValue::by_ref(Pointer::new(block_params[0]), arg_abi.layout))
        }
        PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ } => {
            assert_eq!(block_params.len(), 2, "{:?}", block_params);
            Some(CValue::by_ref_unsized(
                Pointer::new(block_params[0]),
                block_params[1],
                arg_abi.layout,
            ))
        }
    }
}
