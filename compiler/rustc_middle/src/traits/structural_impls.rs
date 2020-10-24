use crate::traits;

use std::fmt;

// Structural impls for the structs in `traits`.

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSource<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::ImplSource::UserDefined(ref v) => write!(f, "{:?}", v),

            super::ImplSource::AutoImpl(ref t) => write!(f, "{:?}", t),

            super::ImplSource::Closure(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Generator(ref d) => write!(f, "{:?}", d),

            super::ImplSource::FnPointer(ref d) => write!(f, "({:?})", d),

            super::ImplSource::DiscriminantKind(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Object(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Param(ref n) => write!(f, "ImplSourceParamData({:?})", n),

            super::ImplSource::Builtin(ref d) => write!(f, "{:?}", d),

            super::ImplSource::TraitAlias(ref d) => write!(f, "{:?}", d),
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceUserDefinedData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceUserDefinedData(impl_def_id={:?}, substs={:?}, nested={:?})",
            self.impl_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceGeneratorData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceGeneratorData(generator_def_id={:?}, substs={:?}, nested={:?})",
            self.generator_def_id, self.substs, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceClosureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceClosureData(closure_def_id={:?}, substs={:?}, nested={:?})",
            self.closure_def_id, self.substs, self.nested
        )
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceBuiltinData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImplSourceBuiltinData(nested={:?})", self.nested)
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceAutoImplData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceAutoImplData(trait_def_id={:?}, nested={:?})",
            self.trait_def_id, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceObjectData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceObjectData(upcast={:?}, vtable_base={}, nested={:?})",
            self.upcast_trait_ref, self.vtable_base, self.nested
        )
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceFnPointerData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImplSourceFnPointerData(fn_ty={:?}, nested={:?})", self.fn_ty, self.nested)
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceTraitAliasData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceTraitAliasData(alias_def_id={:?}, substs={:?}, nested={:?})",
            self.alias_def_id, self.substs, self.nested
        )
    }
}

///////////////////////////////////////////////////////////////////////////
// Lift implementations

TrivialTypeFoldableAndLiftImpls! {
    super::IfExpressionCause,
    super::ImplSourceDiscriminantKindData,
}
