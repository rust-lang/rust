use crate::hir::def_id::{DefId, LOCAL_CRATE};
use crate::ich::StableHashingContext;
use rustc_data_structures::stable_hasher::{StableHasher, HashStable,
                                           StableHasherResult};
use std::cmp;
use std::mem;
use crate::ty::{self, TyCtxt};
use crate::ty::subst::SubstsRef;

/// The SymbolExportLevel of a symbols specifies from which kinds of crates
/// the symbol will be exported. `C` symbols will be exported from any
/// kind of crate, including cdylibs which export very few things.
/// `Rust` will only be exported if the crate produced is a Rust
/// dylib.
#[derive(Eq, PartialEq, Debug, Copy, Clone, RustcEncodable, RustcDecodable)]
pub enum SymbolExportLevel {
    C,
    Rust,
}

impl_stable_hash_for!(enum self::SymbolExportLevel {
    C,
    Rust
});

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
    pub fn symbol_name(&self, tcx: TyCtxt<'tcx>) -> ty::SymbolName {
        match *self {
            ExportedSymbol::NonGeneric(def_id) => {
                tcx.symbol_name(ty::Instance::mono(tcx, def_id))
            }
            ExportedSymbol::Generic(def_id, substs) => {
                tcx.symbol_name(ty::Instance::new(def_id, substs))
            }
            ExportedSymbol::NoDefId(symbol_name) => {
                symbol_name
            }
        }
    }

    pub fn compare_stable(&self, tcx: TyCtxt<'tcx>, other: &ExportedSymbol<'tcx>) -> cmp::Ordering {
        match *self {
            ExportedSymbol::NonGeneric(self_def_id) => match *other {
                ExportedSymbol::NonGeneric(other_def_id) => {
                    tcx.def_path_hash(self_def_id).cmp(&tcx.def_path_hash(other_def_id))
                }
                ExportedSymbol::Generic(..) |
                ExportedSymbol::NoDefId(_) => {
                    cmp::Ordering::Less
                }
            }
            ExportedSymbol::Generic(..) => match *other {
                ExportedSymbol::NonGeneric(_) => {
                    cmp::Ordering::Greater
                }
                ExportedSymbol::Generic(..) => {
                    self.symbol_name(tcx).cmp(&other.symbol_name(tcx))
                }
                ExportedSymbol::NoDefId(_) => {
                    cmp::Ordering::Less
                }
            }
            ExportedSymbol::NoDefId(self_symbol_name) => match *other {
                ExportedSymbol::NonGeneric(_) |
                ExportedSymbol::Generic(..) => {
                    cmp::Ordering::Greater
                }
                ExportedSymbol::NoDefId(ref other_symbol_name) => {
                    self_symbol_name.cmp(other_symbol_name)
                }
            }
        }
    }
}

pub fn metadata_symbol_name(tcx: TyCtxt<'_>) -> String {
    format!("rust_metadata_{}_{}",
            tcx.original_crate_name(LOCAL_CRATE),
            tcx.crate_disambiguator(LOCAL_CRATE).to_fingerprint().to_hex())
}

impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ExportedSymbol<'tcx> {
    fn hash_stable<W: StableHasherResult>(&self,
                                          hcx: &mut StableHashingContext<'a>,
                                          hasher: &mut StableHasher<W>) {
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
