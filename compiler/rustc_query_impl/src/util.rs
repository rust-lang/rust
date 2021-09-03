use rustc_hir::def::DefKind;
use rustc_query_system::query::SimpleDefKind;

/// Convert a [`DefKind`] to a [`SimpleDefKind`].
///
/// *See [`SimpleDefKind`]'s docs for more information.*
pub(crate) fn def_kind_to_simple_def_kind(def_kind: DefKind) -> SimpleDefKind {
    match def_kind {
        DefKind::Struct => SimpleDefKind::Struct,
        DefKind::Enum => SimpleDefKind::Enum,
        DefKind::Union => SimpleDefKind::Union,
        DefKind::Trait => SimpleDefKind::Trait,
        DefKind::TyAlias => SimpleDefKind::TyAlias,
        DefKind::TraitAlias => SimpleDefKind::TraitAlias,

        _ => SimpleDefKind::Other,
    }
}
