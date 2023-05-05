use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_query_system::ich::StableHashingContext;

use crate::ty::{
    self, AliasKind, AliasTy, Const, ConstKind, Decodable, Encodable, FallibleTypeFolder,
    GenericArg, Interned, ParamConst, ParamTy, Ty, TyCtxt, TyDecoder, TyEncoder, TypeFoldable,
    TypeVisitable, TypeVisitor, WithCachedTypeInfo, CONST_TAG, TAG_MASK, TYPE_TAG,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Term<'tcx> {
    ptr: NonZeroUsize,
    marker: PhantomData<(Ty<'tcx>, Const<'tcx>)>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub enum TermKind<'tcx> {
    Ty(Ty<'tcx>),
    Const(Const<'tcx>),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ParamTerm {
    Ty(ParamTy),
    Const(ParamConst),
}

impl<'tcx> Term<'tcx> {
    #[inline]
    pub fn unpack(self) -> TermKind<'tcx> {
        let ptr = self.ptr.get();
        // SAFETY: use of `Interned::new_unchecked` here is ok because these
        // pointers were originally created from `Interned` types in `pack()`,
        // and this is just going in the other direction.
        unsafe {
            match ptr & TAG_MASK {
                TYPE_TAG => TermKind::Ty(Ty(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const WithCachedTypeInfo<ty::TyKind<'tcx>>),
                ))),
                CONST_TAG => TermKind::Const(ty::Const(Interned::new_unchecked(
                    &*((ptr & !TAG_MASK) as *const ty::ConstData<'tcx>),
                ))),
                _ => core::intrinsics::unreachable(),
            }
        }
    }

    pub fn ty(&self) -> Option<Ty<'tcx>> {
        if let TermKind::Ty(ty) = self.unpack() { Some(ty) } else { None }
    }

    pub fn ct(&self) -> Option<Const<'tcx>> {
        if let TermKind::Const(c) = self.unpack() { Some(c) } else { None }
    }

    pub fn into_arg(self) -> GenericArg<'tcx> {
        match self.unpack() {
            TermKind::Ty(ty) => ty.into(),
            TermKind::Const(c) => c.into(),
        }
    }

    /// This function returns the inner `AliasTy` if this term is a projection.
    ///
    /// FIXME: rename `AliasTy` to `AliasTerm` and make sure we correctly
    /// deal with constants.
    pub fn to_projection_term(&self, tcx: TyCtxt<'tcx>) -> Option<AliasTy<'tcx>> {
        match self.unpack() {
            TermKind::Ty(ty) => match ty.kind() {
                ty::Alias(kind, alias_ty) => match kind {
                    AliasKind::Projection => Some(*alias_ty),
                    AliasKind::Opaque => None,
                },
                _ => None,
            },
            TermKind::Const(ct) => match ct.kind() {
                ConstKind::Unevaluated(uv) => Some(tcx.mk_alias_ty(uv.def, uv.substs)),
                _ => None,
            },
        }
    }

    pub fn is_infer(&self) -> bool {
        match self.unpack() {
            TermKind::Ty(ty) => ty.is_ty_var(),
            TermKind::Const(ct) => ct.is_ct_infer(),
        }
    }
}

impl fmt::Debug for Term<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = if let Some(ty) = self.ty() {
            format!("Term::Ty({:?})", ty)
        } else if let Some(ct) = self.ct() {
            format!("Term::Ct({:?})", ct)
        } else {
            unreachable!()
        };
        f.write_str(&data)
    }
}

impl<'tcx> From<Ty<'tcx>> for Term<'tcx> {
    fn from(ty: Ty<'tcx>) -> Self {
        TermKind::Ty(ty).pack()
    }
}

impl<'tcx> From<Const<'tcx>> for Term<'tcx> {
    fn from(c: Const<'tcx>) -> Self {
        TermKind::Const(c).pack()
    }
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for Term<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        self.unpack().hash_stable(hcx, hasher);
    }
}

impl<'tcx> TypeFoldable<TyCtxt<'tcx>> for Term<'tcx> {
    fn try_fold_with<F: FallibleTypeFolder<TyCtxt<'tcx>>>(
        self,
        folder: &mut F,
    ) -> Result<Self, F::Error> {
        Ok(self.unpack().try_fold_with(folder)?.pack())
    }
}

impl<'tcx> TypeVisitable<TyCtxt<'tcx>> for Term<'tcx> {
    fn visit_with<V: TypeVisitor<TyCtxt<'tcx>>>(&self, visitor: &mut V) -> ControlFlow<V::BreakTy> {
        self.unpack().visit_with(visitor)
    }
}

impl<'tcx, E: TyEncoder<I = TyCtxt<'tcx>>> Encodable<E> for Term<'tcx> {
    fn encode(&self, e: &mut E) {
        self.unpack().encode(e)
    }
}

impl<'tcx, D: TyDecoder<I = TyCtxt<'tcx>>> Decodable<D> for Term<'tcx> {
    fn decode(d: &mut D) -> Self {
        let res: TermKind<'tcx> = Decodable::decode(d);
        res.pack()
    }
}

impl<'tcx> TermKind<'tcx> {
    #[inline]
    pub(super) fn pack(self) -> Term<'tcx> {
        let (tag, ptr) = match self {
            TermKind::Ty(ty) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ty.0.0) & TAG_MASK, 0);
                (TYPE_TAG, ty.0.0 as *const WithCachedTypeInfo<ty::TyKind<'tcx>> as usize)
            }
            TermKind::Const(ct) => {
                // Ensure we can use the tag bits.
                assert_eq!(mem::align_of_val(&*ct.0.0) & TAG_MASK, 0);
                (CONST_TAG, ct.0.0 as *const ty::ConstData<'tcx> as usize)
            }
        };

        Term { ptr: unsafe { NonZeroUsize::new_unchecked(ptr | tag) }, marker: PhantomData }
    }
}

impl ParamTerm {
    pub fn index(self) -> usize {
        match self {
            ParamTerm::Ty(ty) => ty.index as usize,
            ParamTerm::Const(ct) => ct.index as usize,
        }
    }
}
