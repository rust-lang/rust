use std::fmt::{self, Debug, Display, Formatter};

use rustc_hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{self as hir};
use rustc_span::Span;
use rustc_target::abi::Size;

use crate::mir::interpret::{ConstValue, ErrorHandled, GlobalAlloc, Scalar};
use crate::mir::{interpret, pretty_print_const_value, Promoted};
use crate::ty::{self, print::pretty_print_const, List, Ty, TyCtxt};
use crate::ty::{GenericArgs, GenericArgsRef};
use crate::ty::{ScalarInt, UserTypeAnnotationIndex};

///////////////////////////////////////////////////////////////////////////
/// Constants
///
/// Two constants are equal if they are the same constant. Note that
/// this does not necessarily mean that they are `==` in Rust. In
/// particular, one must be wary of `NaN`!

#[derive(Clone, Copy, PartialEq, TyEncodable, TyDecodable, Hash, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct Constant<'tcx> {
    pub span: Span,

    /// Optional user-given type: for something like
    /// `collect::<Vec<_>>`, this would be present and would
    /// indicate that `Vec<_>` was explicitly specified.
    ///
    /// Needed for NLL to impose user-given type constraints.
    pub user_ty: Option<UserTypeAnnotationIndex>,

    pub literal: ConstantKind<'tcx>,
}

#[derive(Clone, Copy, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable, Debug)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum ConstantKind<'tcx> {
    /// This constant came from the type system.
    ///
    /// Any way of turning `ty::Const` into `ConstValue` should go through `valtree_to_const_val`;
    /// this ensures that we consistently produce "clean" values without data in the padding or
    /// anything like that.
    Ty(ty::Const<'tcx>),

    /// An unevaluated mir constant which is not part of the type system.
    Unevaluated(UnevaluatedConst<'tcx>, Ty<'tcx>),

    /// This constant cannot go back into the type system, as it represents
    /// something the type system cannot handle (e.g. pointers).
    Val(interpret::ConstValue<'tcx>, Ty<'tcx>),
}

impl<'tcx> Constant<'tcx> {
    pub fn check_static_ptr(&self, tcx: TyCtxt<'_>) -> Option<DefId> {
        match self.literal.try_to_scalar() {
            Some(Scalar::Ptr(ptr, _size)) => match tcx.global_alloc(ptr.provenance) {
                GlobalAlloc::Static(def_id) => {
                    assert!(!tcx.is_thread_local_static(def_id));
                    Some(def_id)
                }
                _ => None,
            },
            _ => None,
        }
    }
    #[inline]
    pub fn ty(&self) -> Ty<'tcx> {
        self.literal.ty()
    }
}

impl<'tcx> ConstantKind<'tcx> {
    #[inline(always)]
    pub fn ty(&self) -> Ty<'tcx> {
        match self {
            ConstantKind::Ty(c) => c.ty(),
            ConstantKind::Val(_, ty) | ConstantKind::Unevaluated(_, ty) => *ty,
        }
    }

    #[inline]
    pub fn try_to_scalar(self) -> Option<Scalar> {
        match self {
            ConstantKind::Ty(c) => match c.kind() {
                ty::ConstKind::Value(valtree) => match valtree {
                    ty::ValTree::Leaf(scalar_int) => Some(Scalar::Int(scalar_int)),
                    ty::ValTree::Branch(_) => None,
                },
                _ => None,
            },
            ConstantKind::Val(val, _) => val.try_to_scalar(),
            ConstantKind::Unevaluated(..) => None,
        }
    }

    #[inline]
    pub fn try_to_scalar_int(self) -> Option<ScalarInt> {
        self.try_to_scalar()?.try_to_int().ok()
    }

    #[inline]
    pub fn try_to_bits(self, size: Size) -> Option<u128> {
        self.try_to_scalar_int()?.to_bits(size).ok()
    }

    #[inline]
    pub fn try_to_bool(self) -> Option<bool> {
        self.try_to_scalar_int()?.try_into().ok()
    }

    #[inline]
    pub fn eval(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        span: Option<Span>,
    ) -> Result<interpret::ConstValue<'tcx>, ErrorHandled> {
        match self {
            ConstantKind::Ty(c) => {
                // We want to consistently have a "clean" value for type system constants (i.e., no
                // data hidden in the padding), so we always go through a valtree here.
                let val = c.eval(tcx, param_env, span)?;
                Ok(tcx.valtree_to_const_val((self.ty(), val)))
            }
            ConstantKind::Unevaluated(uneval, _) => {
                // FIXME: We might want to have a `try_eval`-like function on `Unevaluated`
                tcx.const_eval_resolve(param_env, uneval, span)
            }
            ConstantKind::Val(val, _) => Ok(val),
        }
    }

    /// Normalizes the constant to a value or an error if possible.
    #[inline]
    pub fn normalize(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Self {
        match self.eval(tcx, param_env, None) {
            Ok(val) => Self::Val(val, self.ty()),
            Err(ErrorHandled::Reported(guar, _span)) => {
                Self::Ty(ty::Const::new_error(tcx, guar.into(), self.ty()))
            }
            Err(ErrorHandled::TooGeneric(_span)) => self,
        }
    }

    #[inline]
    pub fn try_eval_scalar(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<Scalar> {
        self.eval(tcx, param_env, None).ok()?.try_to_scalar()
    }

    #[inline]
    pub fn try_eval_scalar_int(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<ScalarInt> {
        self.try_eval_scalar(tcx, param_env)?.try_to_int().ok()
    }

    #[inline]
    pub fn try_eval_bits(
        &self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        ty: Ty<'tcx>,
    ) -> Option<u128> {
        let int = self.try_eval_scalar_int(tcx, param_env)?;
        assert_eq!(self.ty(), ty);
        let size = tcx.layout_of(param_env.with_reveal_all_normalized(tcx).and(ty)).ok()?.size;
        int.to_bits(size).ok()
    }

    /// Panics if the value cannot be evaluated or doesn't contain a valid integer of the given type.
    #[inline]
    pub fn eval_bits(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>, ty: Ty<'tcx>) -> u128 {
        self.try_eval_bits(tcx, param_env, ty)
            .unwrap_or_else(|| bug!("expected bits of {:#?}, got {:#?}", ty, self))
    }

    #[inline]
    pub fn try_eval_target_usize(
        self,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Option<u64> {
        self.try_eval_scalar_int(tcx, param_env)?.try_to_target_usize(tcx).ok()
    }

    #[inline]
    /// Panics if the value cannot be evaluated or doesn't contain a valid `usize`.
    pub fn eval_target_usize(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> u64 {
        self.try_eval_target_usize(tcx, param_env)
            .unwrap_or_else(|| bug!("expected usize, got {:#?}", self))
    }

    #[inline]
    pub fn try_eval_bool(self, tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Option<bool> {
        self.try_eval_scalar_int(tcx, param_env)?.try_into().ok()
    }

    #[inline]
    pub fn from_value(val: ConstValue<'tcx>, ty: Ty<'tcx>) -> Self {
        Self::Val(val, ty)
    }

    pub fn from_bits(
        tcx: TyCtxt<'tcx>,
        bits: u128,
        param_env_ty: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Self {
        let size = tcx
            .layout_of(param_env_ty)
            .unwrap_or_else(|e| {
                bug!("could not compute layout for {:?}: {:?}", param_env_ty.value, e)
            })
            .size;
        let cv = ConstValue::Scalar(Scalar::from_uint(bits, size));

        Self::Val(cv, param_env_ty.value)
    }

    #[inline]
    pub fn from_bool(tcx: TyCtxt<'tcx>, v: bool) -> Self {
        let cv = ConstValue::from_bool(v);
        Self::Val(cv, tcx.types.bool)
    }

    #[inline]
    pub fn zero_sized(ty: Ty<'tcx>) -> Self {
        let cv = ConstValue::ZeroSized;
        Self::Val(cv, ty)
    }

    pub fn from_usize(tcx: TyCtxt<'tcx>, n: u64) -> Self {
        let ty = tcx.types.usize;
        Self::from_bits(tcx, n as u128, ty::ParamEnv::empty().and(ty))
    }

    #[inline]
    pub fn from_scalar(_tcx: TyCtxt<'tcx>, s: Scalar, ty: Ty<'tcx>) -> Self {
        let val = ConstValue::Scalar(s);
        Self::Val(val, ty)
    }

    /// Literals are converted to `ConstantKindVal`, const generic parameters are eagerly
    /// converted to a constant, everything else becomes `Unevaluated`.
    #[instrument(skip(tcx), level = "debug", ret)]
    pub fn from_anon_const(
        tcx: TyCtxt<'tcx>,
        def: LocalDefId,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        let body_id = match tcx.hir().get_by_def_id(def) {
            hir::Node::AnonConst(ac) => ac.body,
            _ => {
                span_bug!(tcx.def_span(def), "from_anon_const can only process anonymous constants")
            }
        };

        let expr = &tcx.hir().body(body_id).value;
        debug!(?expr);

        // Unwrap a block, so that e.g. `{ P }` is recognised as a parameter. Const arguments
        // currently have to be wrapped in curly brackets, so it's necessary to special-case.
        let expr = match &expr.kind {
            hir::ExprKind::Block(block, _) if block.stmts.is_empty() && block.expr.is_some() => {
                block.expr.as_ref().unwrap()
            }
            _ => expr,
        };
        debug!("expr.kind: {:?}", expr.kind);

        let ty = tcx.type_of(def).instantiate_identity();
        debug!(?ty);

        // FIXME(const_generics): We currently have to special case parameters because `min_const_generics`
        // does not provide the parents generics to anonymous constants. We still allow generic const
        // parameters by themselves however, e.g. `N`. These constants would cause an ICE if we were to
        // ever try to substitute the generic parameters in their bodies.
        //
        // While this doesn't happen as these constants are always used as `ty::ConstKind::Param`, it does
        // cause issues if we were to remove that special-case and try to evaluate the constant instead.
        use hir::{def::DefKind::ConstParam, def::Res, ExprKind, Path, QPath};
        match expr.kind {
            ExprKind::Path(QPath::Resolved(_, &Path { res: Res::Def(ConstParam, def_id), .. })) => {
                // Find the name and index of the const parameter by indexing the generics of
                // the parent item and construct a `ParamConst`.
                let item_def_id = tcx.parent(def_id);
                let generics = tcx.generics_of(item_def_id);
                let index = generics.param_def_id_to_index[&def_id];
                let name = tcx.item_name(def_id);
                let ty_const = ty::Const::new_param(tcx, ty::ParamConst::new(index, name), ty);
                debug!(?ty_const);

                return Self::Ty(ty_const);
            }
            _ => {}
        }

        let hir_id = tcx.hir().local_def_id_to_hir_id(def);
        let parent_args = if let Some(parent_hir_id) = tcx.hir().opt_parent_id(hir_id)
            && let Some(parent_did) = parent_hir_id.as_owner()
        {
            GenericArgs::identity_for_item(tcx, parent_did)
        } else {
            List::empty()
        };
        debug!(?parent_args);

        let did = def.to_def_id();
        let child_args = GenericArgs::identity_for_item(tcx, did);
        let args = tcx.mk_args_from_iter(parent_args.into_iter().chain(child_args.into_iter()));
        debug!(?args);

        let span = tcx.def_span(def);
        let uneval = UnevaluatedConst::new(did, args);
        debug!(?span, ?param_env);

        match tcx.const_eval_resolve(param_env, uneval, Some(span)) {
            Ok(val) => {
                debug!("evaluated const value");
                Self::Val(val, ty)
            }
            Err(_) => {
                debug!("error encountered during evaluation");
                // Error was handled in `const_eval_resolve`. Here we just create a
                // new unevaluated const and error hard later in codegen
                Self::Unevaluated(
                    UnevaluatedConst {
                        def: did,
                        args: GenericArgs::identity_for_item(tcx, did),
                        promoted: None,
                    },
                    ty,
                )
            }
        }
    }

    pub fn from_ty_const(c: ty::Const<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        match c.kind() {
            ty::ConstKind::Value(valtree) => {
                // Make sure that if `c` is normalized, then the return value is normalized.
                let const_val = tcx.valtree_to_const_val((c.ty(), valtree));
                Self::Val(const_val, c.ty())
            }
            _ => Self::Ty(c),
        }
    }
}

/// An unevaluated (potentially generic) constant used in MIR.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(Hash, HashStable, TypeFoldable, TypeVisitable)]
pub struct UnevaluatedConst<'tcx> {
    pub def: DefId,
    pub args: GenericArgsRef<'tcx>,
    pub promoted: Option<Promoted>,
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn shrink(self) -> ty::UnevaluatedConst<'tcx> {
        assert_eq!(self.promoted, None);
        ty::UnevaluatedConst { def: self.def, args: self.args }
    }
}

impl<'tcx> UnevaluatedConst<'tcx> {
    #[inline]
    pub fn new(def: DefId, args: GenericArgsRef<'tcx>) -> UnevaluatedConst<'tcx> {
        UnevaluatedConst { def, args, promoted: Default::default() }
    }

    #[inline]
    pub fn from_instance(instance: ty::Instance<'tcx>) -> Self {
        UnevaluatedConst::new(instance.def_id(), instance.args)
    }
}

impl<'tcx> Debug for Constant<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        write!(fmt, "{self}")
    }
}

impl<'tcx> Display for Constant<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match self.ty().kind() {
            ty::FnDef(..) => {}
            _ => write!(fmt, "const ")?,
        }
        Display::fmt(&self.literal, fmt)
    }
}

impl<'tcx> Display for ConstantKind<'tcx> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            ConstantKind::Ty(c) => pretty_print_const(c, fmt, true),
            ConstantKind::Val(val, ty) => pretty_print_const_value(val, ty, fmt),
            // FIXME(valtrees): Correctly print mir constants.
            ConstantKind::Unevaluated(..) => {
                fmt.write_str("_")?;
                Ok(())
            }
        }
    }
}
