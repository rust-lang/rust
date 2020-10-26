//! Argument passing

use crate::prelude::*;

pub(super) use EmptySinglePair::*;

#[derive(Copy, Clone, Debug)]
pub(super) enum PassMode {
    NoPass,
    ByVal(Type),
    ByValPair(Type, Type),
    ByRef { size: Option<Size> },
}

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

impl PassMode {
    pub(super) fn get_param_ty(self, tcx: TyCtxt<'_>) -> EmptySinglePair<Type> {
        match self {
            PassMode::NoPass => Empty,
            PassMode::ByVal(clif_type) => Single(clif_type),
            PassMode::ByValPair(a, b) => Pair(a, b),
            PassMode::ByRef { size: Some(_) } => Single(pointer_ty(tcx)),
            PassMode::ByRef { size: None } => Pair(pointer_ty(tcx), pointer_ty(tcx)),
        }
    }
}

pub(super) fn get_pass_mode<'tcx>(tcx: TyCtxt<'tcx>, layout: TyAndLayout<'tcx>) -> PassMode {
    if layout.is_zst() {
        // WARNING zst arguments must never be passed, as that will break CastKind::ClosureFnPointer
        PassMode::NoPass
    } else {
        match &layout.abi {
            Abi::Uninhabited => PassMode::NoPass,
            Abi::Scalar(scalar) => PassMode::ByVal(scalar_to_clif_type(tcx, scalar.clone())),
            Abi::ScalarPair(a, b) => {
                let a = scalar_to_clif_type(tcx, a.clone());
                let b = scalar_to_clif_type(tcx, b.clone());
                if a == types::I128 && b == types::I128 {
                    // Returning (i128, i128) by-val-pair would take 4 regs, while only 3 are
                    // available on x86_64. Cranelift gets confused when too many return params
                    // are used.
                    PassMode::ByRef {
                        size: Some(layout.size),
                    }
                } else {
                    PassMode::ByValPair(a, b)
                }
            }

            // FIXME implement Vector Abi in a cg_llvm compatible way
            Abi::Vector { .. } => {
                if let Some(vector_ty) = crate::intrinsics::clif_vector_type(tcx, layout) {
                    PassMode::ByVal(vector_ty)
                } else {
                    PassMode::ByRef {
                        size: Some(layout.size),
                    }
                }
            }

            Abi::Aggregate { sized: true } => PassMode::ByRef {
                size: Some(layout.size),
            },
            Abi::Aggregate { sized: false } => PassMode::ByRef { size: None },
        }
    }
}

/// Get a set of values to be passed as function arguments.
pub(super) fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Module>,
    arg: CValue<'tcx>,
) -> EmptySinglePair<Value> {
    match get_pass_mode(fx.tcx, arg.layout()) {
        PassMode::NoPass => Empty,
        PassMode::ByVal(_) => Single(arg.load_scalar(fx)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            Pair(a, b)
        }
        PassMode::ByRef { size: _ } => match arg.force_stack(fx) {
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
    let pass_mode = get_pass_mode(fx.tcx, layout);

    if let PassMode::NoPass = pass_mode {
        return None;
    }

    let clif_types = pass_mode.get_param_ty(fx.tcx);
    let block_params = clif_types.map(|t| fx.bcx.append_block_param(start_block, t));

    #[cfg(debug_assertions)]
    crate::abi::comments::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        block_params,
        pass_mode,
        arg_ty,
    );

    match pass_mode {
        PassMode::NoPass => unreachable!(),
        PassMode::ByVal(_) => Some(CValue::by_val(block_params.assert_single(), layout)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = block_params.assert_pair();
            Some(CValue::by_val_pair(a, b, layout))
        }
        PassMode::ByRef { size: Some(_) } => Some(CValue::by_ref(
            Pointer::new(block_params.assert_single()),
            layout,
        )),
        PassMode::ByRef { size: None } => {
            let (ptr, meta) = block_params.assert_pair();
            Some(CValue::by_ref_unsized(Pointer::new(ptr), meta, layout))
        }
    }
}
