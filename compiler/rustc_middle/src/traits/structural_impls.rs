use crate::traits;

use std::fmt;

// Structural impls for the structs in `traits`.

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSource<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::ImplSource::UserDefined(ref v) => write!(f, "{:?}", v),

            super::ImplSource::Closure(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Generator(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Future(ref d) => write!(f, "{:?}", d),

            super::ImplSource::FnPointer(ref d) => write!(f, "({:?})", d),

            super::ImplSource::Object(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Param(ref n, ct) => {
                write!(f, "ImplSourceParamData({:?}, {:?})", n, ct)
            }

            super::ImplSource::Builtin(ref d) => write!(f, "{:?}", d),

            super::ImplSource::TraitAlias(ref d) => write!(f, "{:?}", d),

            super::ImplSource::TraitUpcasting(ref d) => write!(f, "{:?}", d),
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

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceFutureData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceFutureData(generator_def_id={:?}, substs={:?}, nested={:?})",
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

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceTraitUpcastingData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceTraitUpcastingData(vtable_vptr_slot={:?}, nested={:?})",
            self.vtable_vptr_slot, self.nested
        )
    }
}

impl<N: fmt::Debug> fmt::Debug for traits::ImplSourceObjectData<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceObjectData(upcast={:?}, vtable_base={}, nested={:?})",
            self.upcast_trait_def_id, self.vtable_base, self.nested
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
