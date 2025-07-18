//! Expansion of associated items

use std::mem;

use cfg::CfgOptions;
use hir_expand::{
    AstId, ExpandTo, HirFileId, InFile, Intern, Lookup, MacroCallKind, MacroDefKind,
    mod_path::ModPath,
    name::{AsName, Name},
    span_map::SpanMap,
};
use intern::Interned;
use span::AstIdMap;
use syntax::{
    AstNode,
    ast::{self, HasModuleItem, HasName},
};
use thin_vec::ThinVec;
use triomphe::Arc;

use crate::{
    AssocItemId, AstIdWithPath, ConstLoc, FunctionId, FunctionLoc, ImplId, ItemContainerId,
    ItemLoc, MacroCallId, ModuleId, TraitId, TypeAliasId, TypeAliasLoc,
    attr::Attrs,
    db::DefDatabase,
    macro_call_as_call_id,
    nameres::{
        DefMap, LocalDefMap, MacroSubNs,
        attr_resolution::ResolvedAttr,
        diagnostics::{DefDiagnostic, DefDiagnostics},
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitItems {
    pub items: Box<[(Name, AssocItemId)]>,
    // `ThinVec` as the vec is usually empty anyways
    pub macro_calls: ThinVec<(AstId<ast::Item>, MacroCallId)>,
}

#[salsa::tracked]
impl TraitItems {
    #[inline]
    pub(crate) fn query(db: &dyn DefDatabase, tr: TraitId) -> &TraitItems {
        &Self::query_with_diagnostics(db, tr).0
    }

    #[salsa::tracked(returns(ref))]
    pub fn query_with_diagnostics(
        db: &dyn DefDatabase,
        tr: TraitId,
    ) -> (TraitItems, DefDiagnostics) {
        let ItemLoc { container: module_id, id: ast_id } = tr.lookup(db);

        let collector =
            AssocItemCollector::new(db, module_id, ItemContainerId::TraitId(tr), ast_id.file_id);
        let source = ast_id.with_value(collector.ast_id_map.get(ast_id.value)).to_node(db);
        let (items, macro_calls, diagnostics) = collector.collect(source.assoc_item_list());

        (TraitItems { macro_calls, items }, DefDiagnostics::new(diagnostics))
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

    pub fn assoc_item_by_name(&self, name: &Name) -> Option<AssocItemId> {
        self.items.iter().find_map(|&(ref item_name, item)| match item {
            AssocItemId::FunctionId(_) if item_name == name => Some(item),
            AssocItemId::TypeAliasId(_) if item_name == name => Some(item),
            AssocItemId::ConstId(_) if item_name == name => Some(item),
            _ => None,
        })
    }

    pub fn macro_calls(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.macro_calls.iter().copied()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ImplItems {
    pub items: Box<[(Name, AssocItemId)]>,
    // `ThinVec` as the vec is usually empty anyways
    pub macro_calls: ThinVec<(AstId<ast::Item>, MacroCallId)>,
}

#[salsa::tracked]
impl ImplItems {
    #[salsa::tracked(returns(ref))]
    pub fn of(db: &dyn DefDatabase, id: ImplId) -> (ImplItems, DefDiagnostics) {
        let _p = tracing::info_span!("impl_items_with_diagnostics_query").entered();
        let ItemLoc { container: module_id, id: ast_id } = id.lookup(db);

        let collector =
            AssocItemCollector::new(db, module_id, ItemContainerId::ImplId(id), ast_id.file_id);
        let source = ast_id.with_value(collector.ast_id_map.get(ast_id.value)).to_node(db);
        let (items, macro_calls, diagnostics) = collector.collect(source.assoc_item_list());

        (ImplItems { items, macro_calls }, DefDiagnostics::new(diagnostics))
    }
}

impl ImplItems {
    pub fn macro_calls(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.macro_calls.iter().copied()
    }
}

struct AssocItemCollector<'a> {
    db: &'a dyn DefDatabase,
    module_id: ModuleId,
    def_map: &'a DefMap,
    local_def_map: &'a LocalDefMap,
    ast_id_map: Arc<AstIdMap>,
    span_map: SpanMap,
    cfg_options: &'a CfgOptions,
    file_id: HirFileId,
    diagnostics: Vec<DefDiagnostic>,
    container: ItemContainerId,

    depth: usize,
    items: Vec<(Name, AssocItemId)>,
    macro_calls: ThinVec<(AstId<ast::Item>, MacroCallId)>,
}

impl<'a> AssocItemCollector<'a> {
    fn new(
        db: &'a dyn DefDatabase,
        module_id: ModuleId,
        container: ItemContainerId,
        file_id: HirFileId,
    ) -> Self {
        let (def_map, local_def_map) = module_id.local_def_map(db);
        Self {
            db,
            module_id,
            def_map,
            local_def_map,
            ast_id_map: db.ast_id_map(file_id),
            span_map: db.span_map(file_id),
            cfg_options: module_id.krate.cfg_options(db),
            file_id,
            container,
            items: Vec::new(),

            depth: 0,
            macro_calls: ThinVec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn collect(
        mut self,
        item_list: Option<ast::AssocItemList>,
    ) -> (Box<[(Name, AssocItemId)]>, ThinVec<(AstId<ast::Item>, MacroCallId)>, Vec<DefDiagnostic>)
    {
        if let Some(item_list) = item_list {
            for item in item_list.assoc_items() {
                self.collect_item(item);
            }
        }
        self.macro_calls.shrink_to_fit();
        (self.items.into_boxed_slice(), self.macro_calls, self.diagnostics)
    }

    fn collect_item(&mut self, item: ast::AssocItem) {
        let ast_id = self.ast_id_map.ast_id(&item);
        let attrs = Attrs::new(self.db, &item, self.span_map.as_ref(), self.cfg_options);
        if let Err(cfg) = attrs.is_cfg_enabled(self.cfg_options) {
            self.diagnostics.push(DefDiagnostic::unconfigured_code(
                self.module_id.local_id,
                InFile::new(self.file_id, ast_id.erase()),
                cfg,
                self.cfg_options.clone(),
            ));
            return;
        }
        let ast_id = InFile::new(self.file_id, ast_id.upcast());

        'attrs: for attr in &*attrs {
            let ast_id_with_path = AstIdWithPath { path: attr.path.clone(), ast_id };

            match self.def_map.resolve_attr_macro(
                self.local_def_map,
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
                    self.collect_macro_items(call_id);
                    return;
                }
                Ok(_) => (),
                Err(_) => {
                    self.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                        self.module_id.local_id,
                        MacroCallKind::Attr { ast_id, attr_args: None, invoc_attr_index: attr.id },
                        attr.path().clone(),
                    ));
                }
            }
        }

        self.record_item(item);
    }

    fn record_item(&mut self, item: ast::AssocItem) {
        match item {
            ast::AssocItem::Fn(function) => {
                let Some(name) = function.name() else { return };
                let ast_id = self.ast_id_map.ast_id(&function);
                let def = FunctionLoc {
                    container: self.container,
                    id: InFile::new(self.file_id, ast_id),
                }
                .intern(self.db);
                self.items.push((name.as_name(), def.into()));
            }
            ast::AssocItem::TypeAlias(type_alias) => {
                let Some(name) = type_alias.name() else { return };
                let ast_id = self.ast_id_map.ast_id(&type_alias);
                let def = TypeAliasLoc {
                    container: self.container,
                    id: InFile::new(self.file_id, ast_id),
                }
                .intern(self.db);
                self.items.push((name.as_name(), def.into()));
            }
            ast::AssocItem::Const(konst) => {
                let Some(name) = konst.name() else { return };
                let ast_id = self.ast_id_map.ast_id(&konst);
                let def =
                    ConstLoc { container: self.container, id: InFile::new(self.file_id, ast_id) }
                        .intern(self.db);
                self.items.push((name.as_name(), def.into()));
            }
            ast::AssocItem::MacroCall(call) => {
                let ast_id = self.ast_id_map.ast_id(&call);
                let ast_id = InFile::new(self.file_id, ast_id);
                let Some(path) = call.path() else { return };
                let range = path.syntax().text_range();
                let Some(path) = ModPath::from_src(self.db, path, &mut |range| {
                    self.span_map.span_for_range(range).ctx
                }) else {
                    return;
                };
                let path = Interned::new(path);
                let ctxt = self.span_map.span_for_range(range).ctx;

                let resolver = |path: &_| {
                    self.def_map
                        .resolve_path(
                            self.local_def_map,
                            self.db,
                            self.module_id.local_id,
                            path,
                            crate::item_scope::BuiltinShadowMode::Other,
                            Some(MacroSubNs::Bang),
                        )
                        .0
                        .take_macros()
                        .map(|it| self.db.macro_def(it))
                };
                match macro_call_as_call_id(
                    self.db,
                    ast_id,
                    &path,
                    ctxt,
                    ExpandTo::Items,
                    self.module_id.krate(),
                    resolver,
                    &mut |ptr, call_id| {
                        self.macro_calls.push((ptr.map(|(_, it)| it.upcast()), call_id))
                    },
                ) {
                    // FIXME: Expansion error?
                    Ok(call_id) => match call_id.value {
                        Some(call_id) => {
                            self.macro_calls.push((ast_id.upcast(), call_id));
                            self.collect_macro_items(call_id);
                        }
                        None => (),
                    },
                    Err(_) => {
                        self.diagnostics.push(DefDiagnostic::unresolved_macro_call(
                            self.module_id.local_id,
                            MacroCallKind::FnLike {
                                ast_id,
                                expand_to: ExpandTo::Items,
                                eager: None,
                            },
                            (*path).clone(),
                        ));
                    }
                }
            }
        }
    }

    fn collect_macro_items(&mut self, macro_call_id: MacroCallId) {
        if self.depth > self.def_map.recursion_limit() as usize {
            tracing::warn!("macro expansion is too deep");
            return;
        }

        let (syntax, span_map) = self.db.parse_macro_expansion(macro_call_id).value;
        let old_file_id = mem::replace(&mut self.file_id, macro_call_id.into());
        let old_ast_id_map = mem::replace(&mut self.ast_id_map, self.db.ast_id_map(self.file_id));
        let old_span_map = mem::replace(&mut self.span_map, SpanMap::ExpansionSpanMap(span_map));
        self.depth += 1;

        let items = ast::MacroItems::cast(syntax.syntax_node()).expect("not `MacroItems`");
        for item in items.items() {
            let item = match item {
                ast::Item::Fn(it) => ast::AssocItem::from(it),
                ast::Item::Const(it) => it.into(),
                ast::Item::TypeAlias(it) => it.into(),
                ast::Item::MacroCall(it) => it.into(),
                // FIXME: Should error on disallowed item kinds.
                _ => continue,
            };
            self.collect_item(item);
        }

        self.depth -= 1;
        self.file_id = old_file_id;
        self.ast_id_map = old_ast_id_map;
        self.span_map = old_span_map;
    }
}
