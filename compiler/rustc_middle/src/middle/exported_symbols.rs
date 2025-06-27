use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_macros::{Decodable, Encodable, HashStable, TyDecodable, TyEncodable};

use crate::ty::{self, GenericArgsRef, Ty, TyCtxt};

/// The SymbolExportLevel of a symbols specifies from which kinds of crates
/// the symbol will be exported. `C` symbols will be exported from any
/// kind of crate, including cdylibs which export very few things.
/// `Rust` will only be exported if the crate produced is a Rust
/// dylib.
#[derive(Eq, PartialEq, Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
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

/// Kind of exported symbols.
#[derive(Eq, PartialEq, Debug, Copy, Clone, Encodable, Decodable, HashStable, Hash)]
pub enum SymbolExportKind {
    Text,
    Data,
    Tls,
}

/// The `SymbolExportInfo` of a symbols specifies symbol-related information
/// that is relevant to code generation and linking.
///
/// The difference between `used` and `rustc_std_internal_symbol` is that the
/// former is exported by LTO while the latter isn't.
#[derive(Eq, PartialEq, Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
pub struct SymbolExportInfo {
    pub level: SymbolExportLevel,
    pub kind: SymbolExportKind,
    /// Was the symbol marked as `#[used(compiler)]` or `#[used(linker)]`?
    pub used: bool,
    /// Was the symbol marked as `#[rustc_std_internal_symbol]`?
    pub rustc_std_internal_symbol: bool,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone, TyEncodable, TyDecodable, HashStable)]
pub enum ExportedSymbol<'tcx> {
    NonGeneric(DefId),
    Generic(DefId, GenericArgsRef<'tcx>),
    DropGlue(Ty<'tcx>),
    AsyncDropGlueCtorShim(Ty<'tcx>),
    AsyncDropGlue(DefId, Ty<'tcx>),
    ThreadLocalShim(DefId),
    NoDefId(ty::SymbolName<'tcx>),
}

impl<'tcx> ExportedSymbol<'tcx> {
    /// This is the symbol name of an instance if it is instantiated in the
    /// local crate.
    pub fn symbol_name_for_local_instance(&self, tcx: TyCtxt<'tcx>) -> ty::SymbolName<'tcx> {
        match *self {
            ExportedSymbol::NonGeneric(def_id) => tcx.symbol_name(ty::Instance::mono(tcx, def_id)),
            ExportedSymbol::Generic(def_id, args) => {
                tcx.symbol_name(ty::Instance::new_raw(def_id, args))
            }
            ExportedSymbol::DropGlue(ty) => {
                tcx.symbol_name(ty::Instance::resolve_drop_in_place(tcx, ty))
            }
            ExportedSymbol::AsyncDropGlueCtorShim(ty) => {
                tcx.symbol_name(ty::Instance::resolve_async_drop_in_place(tcx, ty))
            }
            ExportedSymbol::AsyncDropGlue(def_id, ty) => {
                tcx.symbol_name(ty::Instance::resolve_async_drop_in_place_poll(tcx, def_id, ty))
            }
            ExportedSymbol::ThreadLocalShim(def_id) => tcx.symbol_name(ty::Instance {
                def: ty::InstanceKind::ThreadLocalShim(def_id),
                args: ty::GenericArgs::empty(),
            }),
            ExportedSymbol::NoDefId(symbol_name) => symbol_name,
        }
    }
}

pub fn metadata_symbol_name(tcx: TyCtxt<'_>) -> String {
    format!(
        "rust_metadata_{}_{:08x}",
        tcx.crate_name(LOCAL_CRATE),
        tcx.stable_crate_id(LOCAL_CRATE),
    )
}
