use hir_def::{
    db::DefDatabase,
    resolver::{HasResolver, TypeNs},
    type_ref::TypeRef,
    TraitId,
};
use hir_expand::name;

// FIXME: this is wrong, b/c it can't express `trait T: PartialEq<()>`.
// We should return a `TraitREf` here.
fn direct_super_traits(db: &impl DefDatabase, trait_: TraitId) -> Vec<TraitId> {
    let resolver = trait_.resolver(db);
    // returning the iterator directly doesn't easily work because of
    // lifetime problems, but since there usually shouldn't be more than a
    // few direct traits this should be fine (we could even use some kind of
    // SmallVec if performance is a concern)
    db.generic_params(trait_.into())
        .where_predicates
        .iter()
        .filter_map(|pred| match &pred.type_ref {
            TypeRef::Path(p) if p.as_ident() == Some(&name::SELF_TYPE) => pred.bound.as_path(),
            _ => None,
        })
        .filter_map(|path| match resolver.resolve_path_in_type_ns_fully(db, path) {
            Some(TypeNs::TraitId(t)) => Some(t),
            _ => None,
        })
        .collect()
}

/// Returns an iterator over the whole super trait hierarchy (including the
/// trait itself).
pub(crate) fn all_super_traits(db: &impl DefDatabase, trait_: TraitId) -> Vec<TraitId> {
    // we need to take care a bit here to avoid infinite loops in case of cycles
    // (i.e. if we have `trait A: B; trait B: A;`)
    let mut result = vec![trait_];
    let mut i = 0;
    while i < result.len() {
        let t = result[i];
        // yeah this is quadratic, but trait hierarchies should be flat
        // enough that this doesn't matter
        for tt in direct_super_traits(db, t) {
            if !result.contains(&tt) {
                result.push(tt);
            }
        }
        i += 1;
    }
    result
}
