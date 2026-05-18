//! Defines database & queries for name resolution.
use base_db::{Crate, SourceDatabase};
use hir_expand::{
    EditionedFileId, HirFileId, InFile, Lookup, MacroCallId, MacroDefId, MacroDefKind,
    db::ExpandDatabase,
};
use triomphe::Arc;

use crate::{
    AssocItemId, AttrDefId, Macro2Loc, MacroExpander, MacroId, MacroRulesLoc, MacroRulesLocFlags,
    TraitId,
    attrs::AttrFlags,
    item_tree::{ItemTree, file_item_tree},
    nameres::crate_def_map,
    visibility::{self, Visibility},
};

#[query_group::query_group]
pub trait DefDatabase: ExpandDatabase + SourceDatabase {
    /// Whether to expand procedural macros during name resolution.
    #[salsa::input]
    fn expand_proc_attr_macros(&self) -> bool;

    /// Computes an [`ItemTree`] for the given file or macro expansion.
    #[salsa::invoke(file_item_tree)]
    #[salsa::transparent]
    fn file_item_tree(&self, file_id: HirFileId, krate: Crate) -> &ItemTree;

    /// Turns a MacroId into a MacroDefId, describing the macro's definition post name resolution.
    #[salsa::invoke(macro_def)]
    fn macro_def(&self, m: MacroId) -> MacroDefId;

    // region:visibilities

    #[salsa::invoke(visibility::assoc_visibility_query)]
    fn assoc_visibility(&self, def: AssocItemId) -> Visibility;

    // endregion:visibilities

    #[salsa::invoke(crate::lang_item::crate_notable_traits)]
    #[salsa::transparent]
    fn crate_notable_traits(&self, krate: Crate) -> Option<&[TraitId]>;

    #[salsa::invoke(crate_supports_no_std)]
    fn crate_supports_no_std(&self, crate_id: Crate) -> bool;

    #[salsa::invoke(include_macro_invoc)]
    fn include_macro_invoc(&self, crate_id: Crate) -> Arc<[(MacroCallId, EditionedFileId)]>;
}

// return: macro call id and include file id
fn include_macro_invoc(
    db: &dyn DefDatabase,
    krate: Crate,
) -> Arc<[(MacroCallId, EditionedFileId)]> {
    crate_def_map(db, krate)
        .modules
        .values()
        .flat_map(|m| m.scope.iter_macro_invoc())
        .filter_map(|invoc| invoc.1.loc(db).include_file_id(db, *invoc.1).map(|x| (*invoc.1, x)))
        .collect()
}

fn crate_supports_no_std(db: &dyn DefDatabase, crate_id: Crate) -> bool {
    let root_module = crate_def_map(db, crate_id).root_module_id();
    let attrs = AttrFlags::query(db, AttrDefId::ModuleId(root_module));
    attrs.contains(AttrFlags::IS_NO_STD)
}

fn macro_def(db: &dyn DefDatabase, id: MacroId) -> MacroDefId {
    let kind = |expander, file_id, m| {
        let in_file = InFile::new(file_id, m);
        match expander {
            MacroExpander::Declarative { styles } => MacroDefKind::Declarative(in_file, styles),
            MacroExpander::BuiltIn(it) => MacroDefKind::BuiltIn(in_file, it),
            MacroExpander::BuiltInAttr(it) => MacroDefKind::BuiltInAttr(in_file, it),
            MacroExpander::BuiltInDerive(it) => MacroDefKind::BuiltInDerive(in_file, it),
            MacroExpander::BuiltInEager(it) => MacroDefKind::BuiltInEager(in_file, it),
        }
    };

    match id {
        MacroId::Macro2Id(it) => {
            let loc: Macro2Loc = it.lookup(db);

            MacroDefId {
                krate: loc.container.krate(db),
                kind: kind(loc.expander, loc.id.file_id, loc.id.value.upcast()),
                local_inner: false,
                allow_internal_unsafe: loc.allow_internal_unsafe,
                edition: loc.edition,
            }
        }
        MacroId::MacroRulesId(it) => {
            let loc: MacroRulesLoc = it.lookup(db);

            MacroDefId {
                krate: loc.container.krate(db),
                kind: kind(loc.expander, loc.id.file_id, loc.id.value.upcast()),
                local_inner: loc.flags.contains(MacroRulesLocFlags::LOCAL_INNER),
                allow_internal_unsafe: loc
                    .flags
                    .contains(MacroRulesLocFlags::ALLOW_INTERNAL_UNSAFE),
                edition: loc.edition,
            }
        }
        MacroId::ProcMacroId(it) => {
            let loc = it.lookup(db);

            MacroDefId {
                krate: loc.container.krate(db),
                kind: MacroDefKind::ProcMacro(loc.id, loc.expander, loc.kind),
                local_inner: false,
                allow_internal_unsafe: false,
                edition: loc.edition,
            }
        }
    }
}
