use rustc_hir::def_id::DefId;

/// Useful source information about where a desugared associated type for an
/// RPITIT originated from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Encodable, Decodable, HashStable)]
pub enum ImplTraitInTraitData {
    Trait { fn_def_id: DefId, opaque_def_id: DefId },
    Impl { fn_def_id: DefId },
}
