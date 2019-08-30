use crate::prelude::*;

#[derive(Copy, Clone, Debug)]
pub enum PassMode {
    NoPass,
    ByVal(Type),
    ByValPair(Type, Type),
    ByRef,
}

#[derive(Copy, Clone, Debug)]
pub enum EmptySinglePair<T> {
    Empty,
    Single(T),
    Pair(T, T),
}

impl<T> EmptySinglePair<T> {
    pub fn into_iter(self) -> EmptySinglePairIter<T> {
        EmptySinglePairIter(self)
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> EmptySinglePair<U> {
        match self {
            Empty => Empty,
            Single(v) => Single(f(v)),
            Pair(a, b) => Pair(f(a), f(b)),
        }
    }
}

pub struct EmptySinglePairIter<T>(EmptySinglePair<T>);

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
    pub fn assert_single(self) -> T {
        match self {
            Single(v) => v,
            _ => panic!("Called assert_single on {:?}", self)
        }
    }

    pub fn assert_pair(self) -> (T, T) {
        match self {
            Pair(a, b) => (a, b),
            _ => panic!("Called assert_pair on {:?}", self)
        }
    }
}

pub use EmptySinglePair::*;

impl PassMode {
    pub fn get_param_ty(self, tcx: TyCtxt<'_>) -> EmptySinglePair<Type> {
        match self {
            PassMode::NoPass => Empty,
            PassMode::ByVal(clif_type) => Single(clif_type),
            PassMode::ByValPair(a, b) => Pair(a, b),
            PassMode::ByRef => Single(pointer_ty(tcx)),
        }
    }
}

pub fn get_pass_mode<'tcx>(
    tcx: TyCtxt<'tcx>,
    layout: TyLayout<'tcx>,
) -> PassMode {
    assert!(!layout.is_unsized());

    if layout.is_zst() {
        // WARNING zst arguments must never be passed, as that will break CastKind::ClosureFnPointer
        PassMode::NoPass
    } else {
        match &layout.abi {
            layout::Abi::Uninhabited => PassMode::NoPass,
            layout::Abi::Scalar(scalar) => {
                PassMode::ByVal(scalar_to_clif_type(tcx, scalar.clone()))
            }
            layout::Abi::ScalarPair(a, b) => {
                let a = scalar_to_clif_type(tcx, a.clone());
                let b = scalar_to_clif_type(tcx, b.clone());
                if a == types::I128 && b == types::I128 {
                    // Returning (i128, i128) by-val-pair would take 4 regs, while only 3 are
                    // available on x86_64. Cranelift gets confused when too many return params
                    // are used.
                    PassMode::ByRef
                } else {
                    PassMode::ByValPair(a, b)
                }
            }

            // FIXME implement Vector Abi in a cg_llvm compatible way
            layout::Abi::Vector { .. } => PassMode::ByRef,

            layout::Abi::Aggregate { .. } => PassMode::ByRef,
        }
    }
}

pub fn adjust_arg_for_abi<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    arg: CValue<'tcx>,
) -> EmptySinglePair<Value> {
    match get_pass_mode(fx.tcx, arg.layout()) {
        PassMode::NoPass => Empty,
        PassMode::ByVal(_) => Single(arg.load_scalar(fx)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            Pair(a, b)
        }
        PassMode::ByRef => Single(arg.force_stack(fx)),
    }
}

pub fn cvalue_for_param<'tcx>(
    fx: &mut FunctionCx<'_, 'tcx, impl Backend>,
    start_ebb: Ebb,
    local: mir::Local,
    local_field: Option<usize>,
    arg_ty: Ty<'tcx>,
    ssa_flags: crate::analyze::Flags,
) -> Option<CValue<'tcx>> {
    let layout = fx.layout_of(arg_ty);
    let pass_mode = get_pass_mode(fx.tcx, fx.layout_of(arg_ty));

    if let PassMode::NoPass = pass_mode {
        return None;
    }

    let clif_types = pass_mode.get_param_ty(fx.tcx);
    let ebb_params = clif_types.map(|t| fx.bcx.append_ebb_param(start_ebb, t));

    #[cfg(debug_assertions)]
    super::add_arg_comment(
        fx,
        "arg",
        local,
        local_field,
        ebb_params,
        pass_mode,
        ssa_flags,
        arg_ty,
    );

    match pass_mode {
        PassMode::NoPass => unreachable!(),
        PassMode::ByVal(_) => Some(CValue::by_val(ebb_params.assert_single(), layout)),
        PassMode::ByValPair(_, _) => {
            let (a, b) = ebb_params.assert_pair();
            Some(CValue::by_val_pair(a, b, layout))
        }
        PassMode::ByRef => Some(CValue::by_ref(ebb_params.assert_single(), layout)),
    }
}
