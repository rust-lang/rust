//! Argument passing

use crate::prelude::*;

use cranelift_codegen::ir::ArgumentPurpose;
use rustc_target::abi::call::{ArgAbi, ArgAttributes, PassMode as RustcPassMode};
pub(super) use EmptySinglePair::*;

#[derive(Copy, Clone, Debug)]
pub(super) enum EmptySinglePair<T> {
    Empty,
    Single(T),
    Pair(T, T),
}

impl<T> EmptySinglePair<T> {
    pub(super) fn into_iter(self) -> EmptySinglePairIter<T> {
        EmptySinglePairIter(self)
    }

    pub(super) fn map<U>(self, mut f: impl FnMut(T) -> U) -> EmptySinglePair<U> {
        match self {
            Empty => Empty,
            Single(v) => Single(f(v)),
            Pair(a, b) => Pair(f(a), f(b)),
        }
    }
}

pub(super) struct EmptySinglePairIter<T>(EmptySinglePair<T>);

impl<T> Iterator for EmptySinglePairIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        match std::mem::replace(&mut self.0, Empty) {
            Empty => None,
            Single(v) => Some(v),
            Pair(a, b) => {
                self.0 = Single(b);
                Some(a)
            }
        }
    }
}

impl<T: std::fmt::Debug> EmptySinglePair<T> {
    pub(super) fn assert_single(self) -> T {
        match self {
            Single(v) => v,
            _ => panic!("Called assert_single on {:?}", self),
        }
    }

    pub(super) fn assert_pair(self) -> (T, T) {
        match self {
            Pair(a, b) => (a, b),
            _ => panic!("Called assert_pair on {:?}", self),
        }
    }
}

pub(super) trait ArgAbiExt<'tcx> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> EmptySinglePair<AbiParam>;
    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>);
}

impl<'tcx> ArgAbiExt<'tcx> for ArgAbi<'tcx, Ty<'tcx>> {
    fn get_abi_param(&self, tcx: TyCtxt<'tcx>) -> EmptySinglePair<AbiParam> {
        match self.mode {
            RustcPassMode::Ignore => EmptySinglePair::Empty,
            RustcPassMode::Direct(_) => match &self.layout.abi {
                Abi::Scalar(scalar) => {
                    EmptySinglePair::Single(AbiParam::new(scalar_to_clif_type(tcx, scalar.clone())))
                }
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    EmptySinglePair::Single(AbiParam::new(vector_ty))
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            RustcPassMode::Pair(_, _) => match &self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a.clone());
                    let b = scalar_to_clif_type(tcx, b.clone());
                    EmptySinglePair::Pair(AbiParam::new(a), AbiParam::new(b))
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            RustcPassMode::Cast(_) => EmptySinglePair::Single(AbiParam::new(pointer_ty(tcx))),
            RustcPassMode::Indirect {
                attrs: _,
                extra_attrs: None,
                on_stack,
            } => {
                if on_stack {
                    let size = u32::try_from(self.layout.size.bytes()).unwrap();
                    EmptySinglePair::Single(AbiParam::special(
                        pointer_ty(tcx),
                        ArgumentPurpose::StructArgument(size),
                    ))
                } else {
                    EmptySinglePair::Single(AbiParam::new(pointer_ty(tcx)))
                }
            }
            RustcPassMode::Indirect {
                attrs: _,
                extra_attrs: Some(_),
                on_stack,
            } => {
                assert!(!on_stack);
                EmptySinglePair::Pair(
                    AbiParam::new(pointer_ty(tcx)),
                    AbiParam::new(pointer_ty(tcx)),
                )
            }
        }
    }

    fn get_abi_return(&self, tcx: TyCtxt<'tcx>) -> (Option<AbiParam>, Vec<AbiParam>) {
        match self.mode {
            RustcPassMode::Ignore => (None, vec![]),
            RustcPassMode::Direct(_) => match &self.layout.abi {
                Abi::Scalar(scalar) => (
                    None,
                    vec![AbiParam::new(scalar_to_clif_type(
                        tcx,
                        scalar.clone(),
                    ))],
                ),
                // FIXME implement Vector Abi in a cg_llvm compatible way
                Abi::Vector { .. } => {
                    let vector_ty = crate::intrinsics::clif_vector_type(tcx, self.layout).unwrap();
                    (None, vec![AbiParam::new(vector_ty)])
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            RustcPassMode::Pair(_, _) => match &self.layout.abi {
                Abi::ScalarPair(a, b) => {
                    let a = scalar_to_clif_type(tcx, a.clone());
                    let b = scalar_to_clif_type(tcx, b.clone());
                    (
                        None,
                        vec![AbiParam::new(a), AbiParam::new(b)],
                    )
                }
                _ => unreachable!("{:?}", self.layout.abi),
            },
            RustcPassMode::Cast(_) => (
                Some(AbiParam::special(
                    pointer_ty(tcx),
                    ArgumentPurpose::StructReturn,
                )),
                vec![],
            ),
            RustcPassMode::Indirect {
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
            RustcPassMode::Indirect {
                attrs: _,
                extra_attrs: Some(_),
                on_stack: _,
            } => unreachable!("unsized return value"),
        }
    }
}

pub(super) fn get_arg_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    layout: TyAndLayout<'tcx>,
) -> ArgAbi<'tcx, Ty<'tcx>> {
    let mut arg_abi = ArgAbi::new(&tcx, layout, |_, _, _| ArgAttributes::new());
    if layout.is_zst() {
        // WARNING zst arguments must never be passed, as that will break CastKind::ClosureFnPointer
        arg_abi.mode = RustcPassMode::Ignore;
    }
    match arg_abi.mode {
        RustcPassMode::Ignore => {}
        RustcPassMode::Direct(_) => match &arg_abi.layout.abi {
            Abi::Scalar(_) => {}
            // FIXME implement Vector Abi in a cg_llvm compatible way
            Abi::Vector { .. } => {
                if crate::intrinsics::clif_vector_type(tcx, arg_abi.layout).is_none() {
                    arg_abi.mode = RustcPassMode::Indirect {
                        attrs: ArgAttributes::new(),
                        extra_attrs: None,
                        on_stack: false,
                    };
                }
            }
            _ => unreachable!("{:?}", arg_abi.layout.abi),
        },
        RustcPassMode::Pair(_, _) => match &arg_abi.layout.abi {
            Abi::ScalarPair(a, b) => {
                let a = scalar_to_clif_type(tcx, a.clone());
                let b = scalar_to_clif_type(tcx, b.clone());
                if a == types::I128 && b == types::I128 {
                    arg_abi.mode = RustcPassMode::Indirect {
                        attrs: ArgAttributes::new(),
                        extra_attrs: None,
                        on_stack: false,
                    };
                }
            }
            _ => unreachable!("{:?}", arg_abi.layout.abi),
        },
        _ => {}
    }
    arg_abi
}

/// Get a set of values to be passed as function arguments.
pub(super) fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    arg: CValue<'tcx>,
) -> EmptySinglePair<Value> {
    let arg_abi = get_arg_abi(fx.tcx, arg.layout());
    match arg_abi.mode {
        RustcPassMode::Ignore => Empty,
        RustcPassMode::Direct(_) => Single(arg.load_scalar(fx)),
        RustcPassMode::Pair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            Pair(a, b)
        }
        RustcPassMode::Cast(_) | RustcPassMode::Indirect { .. } => match arg.force_stack(fx) {
            (ptr, None) => Single(ptr.get_addr(fx)),
            (ptr, Some(meta)) => Pair(ptr.get_addr(fx), meta),
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
    arg_ty: Ty<'tcx>,
) -> Option<CValue<'tcx>> {
    let layout = fx.layout_of(arg_ty);
    let arg_abi = get_arg_abi(fx.tcx, layout);

    let clif_types = arg_abi.get_abi_param(fx.tcx);
    let block_params =
        clif_types.map(|abi_param| fx.bcx.append_block_param(start_block, abi_param.value_type));

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        block_params,
        &arg_abi,
        arg_ty,
    );

    match arg_abi.mode {
        RustcPassMode::Ignore => None,
        RustcPassMode::Direct(_) => Some(CValue::by_val(block_params.assert_single(), layout)),
        RustcPassMode::Pair(_, _) => {
            let (a, b) = block_params.assert_pair();
            Some(CValue::by_val_pair(a, b, layout))
        }
        RustcPassMode::Cast(_)
        | RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: None,
            on_stack: _,
        } => Some(CValue::by_ref(
            Pointer::new(block_params.assert_single()),
            layout,
        )),
        RustcPassMode::Indirect {
            attrs: _,
            extra_attrs: Some(_),
            on_stack: _,
        } => {
            let (ptr, meta) = block_params.assert_pair();
            Some(CValue::by_ref_unsized(Pointer::new(ptr), meta, layout))
        }
    }
}
