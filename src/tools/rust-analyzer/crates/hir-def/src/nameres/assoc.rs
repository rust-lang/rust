//! Expansion of associated items

use hir_expand::{
    name::Name, AstId, ExpandResult, InFile, Intern, Lookup, MacroCallKind, MacroDefKind,
};
use smallvec::SmallVec;
use span::{HirFileId, MacroCallId};
use syntax::{ast, Parse};
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    expander::{Expander, Mark},
    item_tree::{self, AssocItem, ItemTree, ItemTreeId, MacroCall, ModItem, TreeId},
    macro_call_as_call_id,
    nameres::{
        attr_resolution::ResolvedAttr,
        diagnostics::{DefDiagnostic, DefDiagnostics},
        DefMap, LocalDefMap, MacroSubNs,
    },
    AssocItemId, AstIdWithPath, ConstLoc, FunctionId, FunctionLoc, ImplId, ItemContainerId,
    ItemLoc, ModuleId, TraitId, TypeAliasId, TypeAliasLoc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitItems {
    pub items: Box<[(Name, AssocItemId)]>,
    // box it as the vec is usually empty anyways
    pub macro_calls: Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
}

impl TraitItems {
    #[inline]
    pub(crate) fn trait_items_query(db: &dyn DefDatabase, tr: TraitId) -> Arc<TraitItems> {
        db.trait_items_with_diagnostics(tr).0
    }

    pub(crate) fn trait_items_with_diagnostics_query(
        db: &dyn DefDatabase,
        tr: TraitId,
    ) -> (Arc<TraitItems>, DefDiagnostics) {
        let ItemLoc { container: module_id, id: tree_id } = tr.lookup(db);
        let item_tree = tree_id.item_tree(db);
        let tr_def = &item_tree[tree_id.value];

        let mut collector =
            AssocItemCollector::new(db, module_id, tree_id.file_id(), ItemContainerId::TraitId(tr));
        collector.collect(&item_tree, tree_id.tree_id(), &tr_def.items);
        let (items, macro_calls, diagnostics) = collector.finish();

        (Arc::new(TraitItems { macro_calls, items }), DefDiagnostics::new(diagnostics))
    }

    pub fn associated_types(&self) -> impl Iterator<Item = TypeAliasId> + '_ {
        self.items.iter().filter_map(|(_name, item)| match item {
            AssocItemId::TypeAliasId(t) => Some(*t),
            _ => None,
        })
    }

    pub fn associated_type_by_name(&self, name: &Name) -> Option<TypeAliasId> {
        self.items.iter().find_map(|(item_name, item)| match item {
            AssocItemId::TypeAliasId(t) if item_name == name => Some(*t),
            _ => None,
        })
    }

    pub fn method_by_name(&self, name: &Name) -> Option<FunctionId> {
        self.items.iter().find_map(|(item_name, item)| match item {
            AssocItemId::FunctionId(t) if item_name == name => Some(*t),
            _ => None,
        })
    }

    pub fn attribute_calls(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.macro_calls.iter().flat_map(|it| it.iter()).copied()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ImplItems {
    pub items: Box<[(Name, AssocItemId)]>,
    // box it as the vec is usually empty anyways
    pub macro_calls: Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
}

impl ImplItems {
    #[inline]
    pub(crate) fn impl_items_query(db: &dyn DefDatabase, id: ImplId) -> Arc<ImplItems> {
        db.impl_items_with_diagnostics(id).0
    }

    pub(crate) fn impl_items_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: ImplId,
    ) -> (Arc<ImplItems>, DefDiagnostics) {
        let _p = tracing::info_span!("impl_items_with_diagnostics_query").entered();
        let ItemLoc { container: module_id, id: tree_id } = id.lookup(db);

        let item_tree = tree_id.item_tree(db);
        let impl_def = &item_tree[tree_id.value];
        let mut collector =
            AssocItemCollector::new(db, module_id, tree_id.file_id(), ItemContainerId::ImplId(id));
        collector.collect(&item_tree, tree_id.tree_id(), &impl_def.items);

        let (items, macro_calls, diagnostics) = collector.finish();

        (Arc::new(ImplItems { items, macro_calls }), DefDiagnostics::new(diagnostics))
    }

    pub fn attribute_calls(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.macro_calls.iter().flat_map(|it| it.iter()).copied()
    }
}

struct AssocItemCollector<'a> {
    db: &'a dyn DefDatabase,
    module_id: ModuleId,
    def_map: Arc<DefMap>,
    local_def_map: Arc<LocalDefMap>,
    diagnostics: Vec<DefDiagnostic>,
    container: ItemContainerId,
    expander: Expander,

    items: Vec<(Name, AssocItemId)>,
    macro_calls: Vec<(AstId<ast::Item>, MacroCallId)>,
}

impl<'a> AssocItemCollector<'a> {
    fn new(
        db: &'a dyn DefDatabase,
        module_id: ModuleId,
        file_id: HirFileId,
        container: ItemContainerId,
    ) -> Self {
        let (def_map, local_def_map) = module_id.local_def_map(db);
        Self {
            db,
            module_id,
            def_map,
            local_def_map,
            container,
            expander: Expander::new(db, file_id, module_id),
            items: Vec::new(),
            macro_calls: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn finish(
        self,
    ) -> (
        Box<[(Name, AssocItemId)]>,
        Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
        Vec<DefDiagnostic>,
    ) {
        (
            self.items.into_boxed_slice(),
            if self.macro_calls.is_empty() { None } else { Some(Box::new(self.macro_calls)) },
            self.diagnostics,
        )
    }

    fn collect(&mut self, item_tree: &ItemTree, tree_id: TreeId, assoc_items: &[AssocItem]) {
        let container = self.container;
        self.items.reserve(assoc_items.len());

        'items: for &item in assoc_items {
            let attrs = item_tree.attrs(self.db, self.module_id.krate, ModItem::from(item).into());
            if !attrs.is_cfg_enabled(self.expander.cfg_options()) {
                self.diagnostics.push(DefDiagnostic::unconfigured_code(
                    self.module_id.local_id,
                    tree_id,
                    ModItem::from(item).into(),
                    attrs.cfg().unwrap(),
                    self.expander.cfg_options().clone(),
                ));
                continue;
            }

            'attrs: for attr in &*attrs {
                let ast_id =
                    AstId::new(self.expander.current_file_id(), item.ast_id(item_tree).upcast());
                let ast_id_with_path = AstIdWithPath { path: attr.path.clone(), ast_id };

                match self.def_map.resolve_attr_macro(
                    &self.local_def_map,
                    self.db,
                    self.module_id.local_id,
                    ast_id_with_path,
                    attr,
                ) {
                    Ok(ResolvedAttr::Macro(call_id)) => {
                        let loc = self.db.lookup_intern_macro_call(call_id);
                        if let MacroDefKind::ProcMacro(_, exp, _) = loc.def.kind {
                            // If there's no expander for the proc macro (e.g. the
                            // proc macro is ignored, or building the proc macro
                            // crate failed), skip expansion like we would if it was
                            // disabled. This is analogous to the handling in
                            // `DefCollector::collect_macros`.
                            if let Some(err) = exp.as_expand_error(self.module_id.krate) {
                                self.diagnostics.push(DefDiagnostic::macro_error(
                                    self.module_id.local_id,
                                    ast_id,
                                    (*attr.path).clone(),
                                    err,
                                ));
                                continue 'attrs;
                            }
                        }

                        self.macro_calls.push((ast_id, call_id));
                        let res =
                            self.expander.enter_expand_id::<ast::MacroItems>(self.db, call_id);
                        self.collect_macro_items(res);
                        continue 'items;
                    }
                    Ok(_) => (),
                    Err(_) => {
                        self.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                            self.module_id.local_id,
                            MacroCallKind::Attr {
                                ast_id,
                                attr_args: None,
                                invoc_attr_index: attr.id,
                            },
                            attr.path().clone(),
                        ));
                    }
                }
            }

            self.collect_item(item_tree, tree_id, container, item);
        }
    }

    fn collect_item(
        &mut self,
        item_tree: &ItemTree,
        tree_id: TreeId,
        container: ItemContainerId,
        item: AssocItem,
    ) {
        match item {
            AssocItem::Function(id) => {
                let item = &item_tree[id];
                let def =
                    FunctionLoc { container, id: ItemTreeId::new(tree_id, id) }.intern(self.db);
                self.items.push((item.name.clone(), def.into()));
            }
            AssocItem::TypeAlias(id) => {
                let item = &item_tree[id];
                let def =
                    TypeAliasLoc { container, id: ItemTreeId::new(tree_id, id) }.intern(self.db);
                self.items.push((item.name.clone(), def.into()));
            }
            AssocItem::Const(id) => {
                let item = &item_tree[id];
                let Some(name) = item.name.clone() else { return };
                let def = ConstLoc { container, id: ItemTreeId::new(tree_id, id) }.intern(self.db);
                self.items.push((name, def.into()));
            }
            AssocItem::MacroCall(call) => {
                let file_id = self.expander.current_file_id();
                let MacroCall { ast_id, expand_to, ctxt, ref path } = item_tree[call];
                let module = self.expander.module.local_id;

                let resolver = |path: &_| {
                    self.def_map
                        .resolve_path(
                            &self.local_def_map,
                            self.db,
                            module,
                            path,
                            crate::item_scope::BuiltinShadowMode::Other,
                            Some(MacroSubNs::Bang),
                        )
                        .0
                        .take_macros()
                        .map(|it| self.db.macro_def(it))
                };
                match macro_call_as_call_id(
                    self.db.upcast(),
                    &AstIdWithPath::new(file_id, ast_id, Clone::clone(path)),
                    ctxt,
                    expand_to,
                    self.expander.krate(),
                    resolver,
                ) {
                    Ok(Some(call_id)) => {
                        let res =
                            self.expander.enter_expand_id::<ast::MacroItems>(self.db, call_id);
                        self.macro_calls.push((InFile::new(file_id, ast_id.upcast()), call_id));
                        self.collect_macro_items(res);
                    }
                    Ok(None) => (),
                    Err(_) => {
                        self.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                            self.module_id.local_id,
                            MacroCallKind::FnLike {
                                ast_id: InFile::new(file_id, ast_id),
                                expand_to,
                                eager: None,
                            },
                            Clone::clone(path),
                        ));
                    }
                }
            }
        }
    }

    fn collect_macro_items(&mut self, res: ExpandResult<Option<(Mark, Parse<ast::MacroItems>)>>) {
        let Some((mark, _parse)) = res.value else { return };

        let tree_id = item_tree::TreeId::new(self.expander.current_file_id(), None);
        let item_tree = tree_id.item_tree(self.db);
        let iter: SmallVec<[_; 2]> =
            item_tree.top_level_items().iter().filter_map(ModItem::as_assoc_item).collect();

        self.collect(&item_tree, tree_id, &iter);

        self.expander.exit(mark);
    }
}
