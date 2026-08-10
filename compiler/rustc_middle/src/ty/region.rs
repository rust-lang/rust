use rustc_hir::def_id::DefId;
use rustc_macros::{StableHash, TyDecodable, TyEncodable};
use rustc_span::{Symbol, kw};
pub use rustc_type_ir::RegionVid;
use rustc_type_ir::{
    LateParamRegion as IrLateParamRegion, Region as IrRegion, RegionKind as IrRegionKind,
};

use crate::ty::{self, BoundVar, TyCtxt};

pub type Region<'tcx> = IrRegion<TyCtxt<'tcx>>;
pub type RegionKind<'tcx> = IrRegionKind<TyCtxt<'tcx>>;
pub type LateParamRegion<'tcx> = IrLateParamRegion<TyCtxt<'tcx>>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(StableHash)]
pub struct EarlyParamRegion {
    pub index: u32,
    pub name: Symbol,
}

impl EarlyParamRegion {
    #[inline]
    pub fn get_name(&self) -> Option<Symbol> {
        if self.is_named() { Some(self.name) } else { None }
    }

    #[inline]
    /// Does this early bound region have a name? Early bound regions normally
    /// always have names except when using anonymous lifetimes (`'_`).
    pub fn is_named(&self) -> bool {
        self.name != kw::UnderscoreLifetime
    }
}

impl rustc_type_ir::inherent::ParamLike for EarlyParamRegion {
    fn index(self) -> u32 {
        self.index
    }
}

impl<'tcx> rustc_type_ir::inherent::RegionName<TyCtxt<'tcx>> for EarlyParamRegion {
    #[inline]
    fn get_name(&self, _tcx: TyCtxt<'tcx>) -> Option<Symbol> {
        self.get_name()
    }

    #[inline]
    /// Does this early bound region have a name? Early bound regions normally
    /// always have names except when using anonymous lifetimes (`'_`).
    fn is_named(&self, _tcx: TyCtxt<'tcx>) -> bool {
        self.is_named()
    }
}

impl std::fmt::Debug for EarlyParamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/#{}", self.name, self.index)
    }
}

/// When liberating bound regions, we map their [`ty::BoundRegionKind`]
/// to this as we need to track the index of anonymous regions. We
/// otherwise end up liberating multiple bound regions to the same
/// late-bound region.
#[derive(Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(StableHash)]
pub enum LateParamRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    ///
    /// Unlike [`ty::BoundRegionKind::Anon`], this tracks the index of the
    /// liberated bound region.
    ///
    /// We should ideally never liberate anonymous regions, but do so for the
    /// sake of diagnostics in `FnCtxt::sig_of_closure_with_expectation`.
    Anon(u32),

    /// An anonymous region parameter with a `Symbol` name.
    ///
    /// Used to give late-bound regions names for things like pretty printing.
    NamedAnon(u32, Symbol),

    /// Late-bound regions that appear in the AST.
    Named(DefId),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    ClosureEnv,
}

impl LateParamRegionKind {
    pub fn from_bound(var: BoundVar, br: ty::BoundRegionKind<'_>) -> LateParamRegionKind {
        match br {
            ty::BoundRegionKind::Anon => LateParamRegionKind::Anon(var.as_u32()),
            ty::BoundRegionKind::Named(def_id) => LateParamRegionKind::Named(def_id),
            ty::BoundRegionKind::ClosureEnv => LateParamRegionKind::ClosureEnv,
            ty::BoundRegionKind::NamedForPrinting(name) => {
                LateParamRegionKind::NamedAnon(var.as_u32(), name)
            }
        }
    }

    pub fn is_named(&self, tcx: TyCtxt<'_>) -> bool {
        self.get_name(tcx).is_some()
    }

    pub fn get_name(&self, tcx: TyCtxt<'_>) -> Option<Symbol> {
        match *self {
            LateParamRegionKind::Named(def_id) => {
                let name = tcx.item_name(def_id);
                if name != kw::UnderscoreLifetime { Some(name) } else { None }
            }
            LateParamRegionKind::NamedAnon(_, name) => Some(name),
            _ => None,
        }
    }

    pub fn get_id(&self) -> Option<DefId> {
        match *self {
            LateParamRegionKind::Named(id) => Some(id),
            _ => None,
        }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(RegionKind<'_>, 20);
    static_assert_size!(ty::WithCachedTypeInfo<RegionKind<'_>>, 28);
    // tidy-alphabetical-end
}
