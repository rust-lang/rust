use crate::traits;

use std::fmt;

// Structural impls for the structs in `traits`.

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSource<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            super::ImplSource::UserDefined(ref v) => write!(f, "{:?}", v),

            super::ImplSource::Builtin(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Object(ref d) => write!(f, "{:?}", d),

            super::ImplSource::Param(ref n, ct) => {
                write!(f, "ImplSourceParamData({:?}, {:?})", n, ct)
            }

            super::ImplSource::TraitUpcasting(ref d) => write!(f, "{:?}", d),
        }
    }
}

impl<'tcx, N: fmt::Debug> fmt::Debug for traits::ImplSourceUserDefinedData<'tcx, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImplSourceUserDefinedData(impl_def_id={:?}, args={:?}, nested={:?})",
            self.impl_def_id, self.args, self.nested
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
            "ImplSourceObjectData(vtable_base={}, nested={:?})",
            self.vtable_base, self.nested
        )
    }
}
