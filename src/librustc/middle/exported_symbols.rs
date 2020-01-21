use crate::ich::StableHashingContext;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, TyCtxt};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use std::mem;

/// The SymbolExportLevel of a symbols specifies from which kinds of crates
/// the symbol will be exported. `C` symbols will be exported from any
/// kind of crate, including cdylibs which export very few things.
/// `Rust` will only be exported if the crate produced is a Rust
/// dylib.
#[derive(Eq, PartialEq, Debug, Copy, Clone, RustcEncodable, RustcDecodable, HashStable)]
pub enum SymbolExportLevel {
    C,
    Rust,
}

impl SymbolExportLevel {
    pub fn is_below_threshold(self, threshold: SymbolExportLevel) -> bool {
        threshold == SymbolExportLevel::Rust // export everything from Rust dylibs
          || self == SymbolExportLevel::C
    }
}

#[derive(Eq, PartialEq, Debug, Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum ExportedSymbol<'tcx> {
    NonGeneric(DefId),
    Generic(DefId, SubstsRef<'tcx>),
    NoDefId(ty::SymbolName),
}

impl<'tcx> ExportedSymbol<'tcx> {
    /// This is the symbol name of an instance if it is instantiated in the
    /// local crate.
    pub fn symbol_name_for_local_instance(&self, tcx: TyCtxt<'tcx>) -> ty::SymbolName {
        match *self {
            ExportedSymbol::NonGeneric(def_id) => tcx.symbol_name(ty::Instance::mono(tcx, def_id)),
            ExportedSymbol::Generic(def_id, substs) => {
                tcx.symbol_name(ty::Instance::new(def_id, substs))
            }
            ExportedSymbol::NoDefId(symbol_name) => symbol_name,
        }
    }
}

pub fn metadata_symbol_name(tcx: TyCtxt<'_>) -> String {
    format!(
        "rust_metadata_{}_{}",
        tcx.original_crate_name(LOCAL_CRATE),
        tcx.crate_disambiguator(LOCAL_CRATE).to_fingerprint().to_hex()
    )
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ExportedSymbol<'tcx> {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        mem::discriminant(self).hash_stable(hcx, hasher);
        match *self {
            ExportedSymbol::NonGeneric(def_id) => {
                def_id.hash_stable(hcx, hasher);
            }
            ExportedSymbol::Generic(def_id, substs) => {
                def_id.hash_stable(hcx, hasher);
                substs.hash_stable(hcx, hasher);
            }
            ExportedSymbol::NoDefId(symbol_name) => {
                symbol_name.hash_stable(hcx, hasher);
            }
        }
    }
}
