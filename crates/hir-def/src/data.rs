//! Contains basic data about various HIR declarations.

use std::sync::Arc;

use hir_expand::{name::Name, AstId, ExpandResult, HirFileId, InFile, MacroCallId, MacroDefKind};
use intern::Interned;
use smallvec::SmallVec;
use syntax::ast;

use crate::{
    attr::Attrs,
    body::{Expander, Mark},
    db::DefDatabase,
    item_tree::{self, AssocItem, FnFlags, ItemTree, ItemTreeId, ModItem, Param, TreeId},
    nameres::{
        attr_resolution::ResolvedAttr,
        diagnostics::DefDiagnostic,
        proc_macro::{parse_macro_name_and_helper_attrs, ProcMacroKind},
        DefMap,
    },
    type_ref::{TraitRef, TypeBound, TypeRef},
    visibility::RawVisibility,
    AssocItemId, AstIdWithPath, ConstId, ConstLoc, FunctionId, FunctionLoc, HasModule, ImplId,
    Intern, ItemContainerId, ItemLoc, Lookup, Macro2Id, MacroRulesId, ModuleId, ProcMacroId,
    StaticId, TraitAliasId, TraitId, TypeAliasId, TypeAliasLoc,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionData {
    pub name: Name,
    pub params: Vec<(Option<Name>, Interned<TypeRef>)>,
    pub ret_type: Interned<TypeRef>,
    pub async_ret_type: Option<Interned<TypeRef>>,
    pub attrs: Attrs,
    pub visibility: RawVisibility,
    pub abi: Option<Interned<str>>,
    pub legacy_const_generics_indices: Box<[u32]>,
    flags: FnFlags,
}

impl FunctionData {
    pub(crate) fn fn_data_query(db: &dyn DefDatabase, func: FunctionId) -> Arc<FunctionData> {
        let loc = func.lookup(db);
        let krate = loc.container.module(db).krate;
        let crate_graph = db.crate_graph();
        let cfg_options = &crate_graph[krate].cfg_options;
        let item_tree = loc.id.item_tree(db);
        let func = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            db.trait_data(trait_id).visibility.clone()
        } else {
            item_tree[func.visibility].clone()
        };

        let enabled_params = func
            .params
            .clone()
            .filter(|&param| item_tree.attrs(db, krate, param.into()).is_cfg_enabled(cfg_options));

        // If last cfg-enabled param is a `...` param, it's a varargs function.
        let is_varargs = enabled_params
            .clone()
            .next_back()
            .map_or(false, |param| matches!(item_tree[param], Param::Varargs));

        let mut flags = func.flags;
        if is_varargs {
            flags |= FnFlags::IS_VARARGS;
        }
        if flags.contains(FnFlags::HAS_SELF_PARAM) {
            // If there's a self param in the syntax, but it is cfg'd out, remove the flag.
            let is_cfgd_out = match func.params.clone().next() {
                Some(param) => {
                    !item_tree.attrs(db, krate, param.into()).is_cfg_enabled(cfg_options)
                }
                None => {
                    stdx::never!("fn HAS_SELF_PARAM but no parameters allocated");
                    true
                }
            };
            if is_cfgd_out {
                cov_mark::hit!(cfgd_out_self_param);
                flags.remove(FnFlags::HAS_SELF_PARAM);
            }
        }

        let legacy_const_generics_indices = item_tree
            .attrs(db, krate, ModItem::from(loc.id.value).into())
            .by_key("rustc_legacy_const_generics")
            .tt_values()
            .next()
            .map(parse_rustc_legacy_const_generics)
            .unwrap_or_default();

        Arc::new(FunctionData {
            name: func.name.clone(),
            params: enabled_params
                .clone()
                .filter_map(|id| match &item_tree[id] {
                    Param::Normal(name, ty) => Some((name.clone(), ty.clone())),
                    Param::Varargs => None,
                })
                .collect(),
            ret_type: func.ret_type.clone(),
            async_ret_type: func.async_ret_type.clone(),
            attrs: item_tree.attrs(db, krate, ModItem::from(loc.id.value).into()),
            visibility,
            abi: func.abi.clone(),
            legacy_const_generics_indices,
            flags,
        })
    }

    pub fn has_body(&self) -> bool {
        self.flags.contains(FnFlags::HAS_BODY)
    }

    /// True if the first param is `self`. This is relevant to decide whether this
    /// can be called as a method.
    pub fn has_self_param(&self) -> bool {
        self.flags.contains(FnFlags::HAS_SELF_PARAM)
    }

    pub fn has_default_kw(&self) -> bool {
        self.flags.contains(FnFlags::HAS_DEFAULT_KW)
    }

    pub fn has_const_kw(&self) -> bool {
        self.flags.contains(FnFlags::HAS_CONST_KW)
    }

    pub fn has_async_kw(&self) -> bool {
        self.flags.contains(FnFlags::HAS_ASYNC_KW)
    }

    pub fn has_unsafe_kw(&self) -> bool {
        self.flags.contains(FnFlags::HAS_UNSAFE_KW)
    }

    pub fn is_varargs(&self) -> bool {
        self.flags.contains(FnFlags::IS_VARARGS)
    }
}

fn parse_rustc_legacy_const_generics(tt: &crate::tt::Subtree) -> Box<[u32]> {
    let mut indices = Vec::new();
    for args in tt.token_trees.chunks(2) {
        match &args[0] {
            tt::TokenTree::Leaf(tt::Leaf::Literal(lit)) => match lit.text.parse() {
                Ok(index) => indices.push(index),
                Err(_) => break,
            },
            _ => break,
        }

        if let Some(comma) = args.get(1) {
            match comma {
                tt::TokenTree::Leaf(tt::Leaf::Punct(punct)) if punct.char == ',' => {}
                _ => break,
            }
        }
    }

    indices.into_boxed_slice()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAliasData {
    pub name: Name,
    pub type_ref: Option<Interned<TypeRef>>,
    pub visibility: RawVisibility,
    pub is_extern: bool,
    pub rustc_has_incoherent_inherent_impls: bool,
    /// Bounds restricting the type alias itself (eg. `type Ty: Bound;` in a trait or impl).
    pub bounds: Vec<Interned<TypeBound>>,
}

impl TypeAliasData {
    pub(crate) fn type_alias_data_query(
        db: &dyn DefDatabase,
        typ: TypeAliasId,
    ) -> Arc<TypeAliasData> {
        let loc = typ.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let typ = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            db.trait_data(trait_id).visibility.clone()
        } else {
            item_tree[typ.visibility].clone()
        };

        let rustc_has_incoherent_inherent_impls = item_tree
            .attrs(db, loc.container.module(db).krate(), ModItem::from(loc.id.value).into())
            .by_key("rustc_has_incoherent_inherent_impls")
            .exists();

        Arc::new(TypeAliasData {
            name: typ.name.clone(),
            type_ref: typ.type_ref.clone(),
            visibility,
            is_extern: matches!(loc.container, ItemContainerId::ExternBlockId(_)),
            rustc_has_incoherent_inherent_impls,
            bounds: typ.bounds.to_vec(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitData {
    pub name: Name,
    pub items: Vec<(Name, AssocItemId)>,
    pub is_auto: bool,
    pub is_unsafe: bool,
    pub rustc_has_incoherent_inherent_impls: bool,
    pub visibility: RawVisibility,
    /// Whether the trait has `#[rust_skip_array_during_method_dispatch]`. `hir_ty` will ignore
    /// method calls to this trait's methods when the receiver is an array and the crate edition is
    /// 2015 or 2018.
    pub skip_array_during_method_dispatch: bool,
    // box it as the vec is usually empty anyways
    pub attribute_calls: Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
}

impl TraitData {
    pub(crate) fn trait_data_query(db: &dyn DefDatabase, tr: TraitId) -> Arc<TraitData> {
        db.trait_data_with_diagnostics(tr).0
    }

    pub(crate) fn trait_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        tr: TraitId,
    ) -> (Arc<TraitData>, Arc<[DefDiagnostic]>) {
        let tr_loc @ ItemLoc { container: module_id, id: tree_id } = tr.lookup(db);
        let item_tree = tree_id.item_tree(db);
        let tr_def = &item_tree[tree_id.value];
        let _cx = stdx::panic_context::enter(format!(
            "trait_data_query({tr:?} -> {tr_loc:?} -> {tr_def:?})"
        ));
        let name = tr_def.name.clone();
        let is_auto = tr_def.is_auto;
        let is_unsafe = tr_def.is_unsafe;
        let visibility = item_tree[tr_def.visibility].clone();
        let attrs = item_tree.attrs(db, module_id.krate(), ModItem::from(tree_id.value).into());
        let skip_array_during_method_dispatch =
            attrs.by_key("rustc_skip_array_during_method_dispatch").exists();
        let rustc_has_incoherent_inherent_impls =
            attrs.by_key("rustc_has_incoherent_inherent_impls").exists();
        let mut collector =
            AssocItemCollector::new(db, module_id, tree_id.file_id(), ItemContainerId::TraitId(tr));
        collector.collect(&item_tree, tree_id.tree_id(), &tr_def.items);
        let (items, attribute_calls, diagnostics) = collector.finish();

        (
            Arc::new(TraitData {
                name,
                attribute_calls,
                items,
                is_auto,
                is_unsafe,
                visibility,
                skip_array_during_method_dispatch,
                rustc_has_incoherent_inherent_impls,
            }),
            diagnostics.into(),
        )
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
        self.attribute_calls.iter().flat_map(|it| it.iter()).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraitAliasData {
    pub name: Name,
    pub visibility: RawVisibility,
}

impl TraitAliasData {
    pub(crate) fn trait_alias_query(db: &dyn DefDatabase, id: TraitAliasId) -> Arc<TraitAliasData> {
        let loc = id.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let alias = &item_tree[loc.id.value];
        let visibility = item_tree[alias.visibility].clone();

        Arc::new(TraitAliasData { name: alias.name.clone(), visibility })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplData {
    pub target_trait: Option<Interned<TraitRef>>,
    pub self_ty: Interned<TypeRef>,
    pub items: Vec<AssocItemId>,
    pub is_negative: bool,
    // box it as the vec is usually empty anyways
    pub attribute_calls: Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
}

impl ImplData {
    pub(crate) fn impl_data_query(db: &dyn DefDatabase, id: ImplId) -> Arc<ImplData> {
        db.impl_data_with_diagnostics(id).0
    }

    pub(crate) fn impl_data_with_diagnostics_query(
        db: &dyn DefDatabase,
        id: ImplId,
    ) -> (Arc<ImplData>, Arc<[DefDiagnostic]>) {
        let _p = profile::span("impl_data_with_diagnostics_query");
        let ItemLoc { container: module_id, id: tree_id } = id.lookup(db);

        let item_tree = tree_id.item_tree(db);
        let impl_def = &item_tree[tree_id.value];
        let target_trait = impl_def.target_trait.clone();
        let self_ty = impl_def.self_ty.clone();
        let is_negative = impl_def.is_negative;

        let mut collector =
            AssocItemCollector::new(db, module_id, tree_id.file_id(), ItemContainerId::ImplId(id));
        collector.collect(&item_tree, tree_id.tree_id(), &impl_def.items);

        let (items, attribute_calls, diagnostics) = collector.finish();
        let items = items.into_iter().map(|(_, item)| item).collect();

        (
            Arc::new(ImplData { target_trait, self_ty, items, is_negative, attribute_calls }),
            diagnostics.into(),
        )
    }

    pub fn attribute_calls(&self) -> impl Iterator<Item = (AstId<ast::Item>, MacroCallId)> + '_ {
        self.attribute_calls.iter().flat_map(|it| it.iter()).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Macro2Data {
    pub name: Name,
    pub visibility: RawVisibility,
    // It's a bit wasteful as currently this is only for builtin `Default` derive macro, but macro2
    // are rarely used in practice so I think it's okay for now.
    /// Derive helpers, if this is a derive rustc_builtin_macro
    pub helpers: Option<Box<[Name]>>,
}

impl Macro2Data {
    pub(crate) fn macro2_data_query(db: &dyn DefDatabase, makro: Macro2Id) -> Arc<Macro2Data> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let helpers = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .by_key("rustc_builtin_macro")
            .tt_values()
            .next()
            .and_then(|attr| parse_macro_name_and_helper_attrs(&attr.token_trees))
            .map(|(_, helpers)| helpers);

        Arc::new(Macro2Data {
            name: makro.name.clone(),
            visibility: item_tree[makro.visibility].clone(),
            helpers,
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroRulesData {
    pub name: Name,
    pub macro_export: bool,
}

impl MacroRulesData {
    pub(crate) fn macro_rules_data_query(
        db: &dyn DefDatabase,
        makro: MacroRulesId,
    ) -> Arc<MacroRulesData> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let macro_export = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .by_key("macro_export")
            .exists();

        Arc::new(MacroRulesData { name: makro.name.clone(), macro_export })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcMacroData {
    pub name: Name,
    /// Derive helpers, if this is a derive
    pub helpers: Option<Box<[Name]>>,
}

impl ProcMacroData {
    pub(crate) fn proc_macro_data_query(
        db: &dyn DefDatabase,
        makro: ProcMacroId,
    ) -> Arc<ProcMacroData> {
        let loc = makro.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let makro = &item_tree[loc.id.value];

        let (name, helpers) = if let Some(def) = item_tree
            .attrs(db, loc.container.krate(), ModItem::from(loc.id.value).into())
            .parse_proc_macro_decl(&makro.name)
        {
            (
                def.name,
                match def.kind {
                    ProcMacroKind::CustomDerive { helpers } => Some(helpers),
                    ProcMacroKind::FnLike | ProcMacroKind::Attr => None,
                },
            )
        } else {
            // eeeh...
            stdx::never!("proc macro declaration is not a proc macro");
            (makro.name.clone(), None)
        };
        Arc::new(ProcMacroData { name, helpers })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstData {
    /// `None` for `const _: () = ();`
    pub name: Option<Name>,
    pub type_ref: Interned<TypeRef>,
    pub visibility: RawVisibility,
}

impl ConstData {
    pub(crate) fn const_data_query(db: &dyn DefDatabase, konst: ConstId) -> Arc<ConstData> {
        let loc = konst.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let konst = &item_tree[loc.id.value];
        let visibility = if let ItemContainerId::TraitId(trait_id) = loc.container {
            db.trait_data(trait_id).visibility.clone()
        } else {
            item_tree[konst.visibility].clone()
        };

        Arc::new(ConstData {
            name: konst.name.clone(),
            type_ref: konst.type_ref.clone(),
            visibility,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticData {
    pub name: Name,
    pub type_ref: Interned<TypeRef>,
    pub visibility: RawVisibility,
    pub mutable: bool,
    pub is_extern: bool,
}

impl StaticData {
    pub(crate) fn static_data_query(db: &dyn DefDatabase, konst: StaticId) -> Arc<StaticData> {
        let loc = konst.lookup(db);
        let item_tree = loc.id.item_tree(db);
        let statik = &item_tree[loc.id.value];

        Arc::new(StaticData {
            name: statik.name.clone(),
            type_ref: statik.type_ref.clone(),
            visibility: item_tree[statik.visibility].clone(),
            mutable: statik.mutable,
            is_extern: matches!(loc.container, ItemContainerId::ExternBlockId(_)),
        })
    }
}

struct AssocItemCollector<'a> {
    db: &'a dyn DefDatabase,
    module_id: ModuleId,
    def_map: Arc<DefMap>,
    inactive_diagnostics: Vec<DefDiagnostic>,
    container: ItemContainerId,
    expander: Expander,

    items: Vec<(Name, AssocItemId)>,
    attr_calls: Vec<(AstId<ast::Item>, MacroCallId)>,
}

impl<'a> AssocItemCollector<'a> {
    fn new(
        db: &'a dyn DefDatabase,
        module_id: ModuleId,
        file_id: HirFileId,
        container: ItemContainerId,
    ) -> Self {
        Self {
            db,
            module_id,
            def_map: module_id.def_map(db),
            container,
            expander: Expander::new(db, file_id, module_id),
            items: Vec::new(),
            attr_calls: Vec::new(),
            inactive_diagnostics: Vec::new(),
        }
    }

    fn finish(
        self,
    ) -> (
        Vec<(Name, AssocItemId)>,
        Option<Box<Vec<(AstId<ast::Item>, MacroCallId)>>>,
        Vec<DefDiagnostic>,
    ) {
        (
            self.items,
            if self.attr_calls.is_empty() { None } else { Some(Box::new(self.attr_calls)) },
            self.inactive_diagnostics,
        )
    }

    // FIXME: proc-macro diagnostics
    fn collect(&mut self, item_tree: &ItemTree, tree_id: TreeId, assoc_items: &[AssocItem]) {
        let container = self.container;
        self.items.reserve(assoc_items.len());

        'items: for &item in assoc_items {
            let attrs = item_tree.attrs(self.db, self.module_id.krate, ModItem::from(item).into());
            if !attrs.is_cfg_enabled(self.expander.cfg_options()) {
                self.inactive_diagnostics.push(DefDiagnostic::unconfigured_code(
                    self.module_id.local_id,
                    InFile::new(self.expander.current_file_id(), item.ast_id(item_tree).upcast()),
                    attrs.cfg().unwrap(),
                    self.expander.cfg_options().clone(),
                ));
                continue;
            }

            'attrs: for attr in &*attrs {
                let ast_id =
                    AstId::new(self.expander.current_file_id(), item.ast_id(item_tree).upcast());
                let ast_id_with_path = AstIdWithPath { path: (*attr.path).clone(), ast_id };

                if let Ok(ResolvedAttr::Macro(call_id)) = self.def_map.resolve_attr_macro(
                    self.db,
                    self.module_id.local_id,
                    ast_id_with_path,
                    attr,
                ) {
                    self.attr_calls.push((ast_id, call_id));
                    // If proc attribute macro expansion is disabled, skip expanding it here
                    if !self.db.enable_proc_attr_macros() {
                        continue 'attrs;
                    }
                    let loc = self.db.lookup_intern_macro_call(call_id);
                    if let MacroDefKind::ProcMacro(exp, ..) = loc.def.kind {
                        // If there's no expander for the proc macro (e.g. the
                        // proc macro is ignored, or building the proc macro
                        // crate failed), skip expansion like we would if it was
                        // disabled. This is analogous to the handling in
                        // `DefCollector::collect_macros`.
                        if exp.is_dummy() {
                            continue 'attrs;
                        }
                    }
                    match self.expander.enter_expand_id::<ast::MacroItems>(self.db, call_id) {
                        ExpandResult { value: Some((mark, _)), .. } => {
                            self.collect_macro_items(mark);
                            continue 'items;
                        }
                        ExpandResult { .. } => {}
                    }
                }
            }

            match item {
                AssocItem::Function(id) => {
                    let item = &item_tree[id];

                    let def =
                        FunctionLoc { container, id: ItemTreeId::new(tree_id, id) }.intern(self.db);
                    self.items.push((item.name.clone(), def.into()));
                }
                AssocItem::Const(id) => {
                    let item = &item_tree[id];

                    let name = match item.name.clone() {
                        Some(name) => name,
                        None => continue,
                    };
                    let def =
                        ConstLoc { container, id: ItemTreeId::new(tree_id, id) }.intern(self.db);
                    self.items.push((name, def.into()));
                }
                AssocItem::TypeAlias(id) => {
                    let item = &item_tree[id];

                    let def = TypeAliasLoc { container, id: ItemTreeId::new(tree_id, id) }
                        .intern(self.db);
                    self.items.push((item.name.clone(), def.into()));
                }
                AssocItem::MacroCall(call) => {
                    if let Some(root) = self.db.parse_or_expand(self.expander.current_file_id()) {
                        let call = &item_tree[call];

                        let ast_id_map = self.db.ast_id_map(self.expander.current_file_id());
                        let call = ast_id_map.get(call.ast_id).to_node(&root);
                        let _cx =
                            stdx::panic_context::enter(format!("collect_items MacroCall: {call}"));
                        let res = self.expander.enter_expand::<ast::MacroItems>(self.db, call);

                        if let Ok(ExpandResult { value: Some((mark, _)), .. }) = res {
                            self.collect_macro_items(mark);
                        }
                    }
                }
            }
        }
    }

    fn collect_macro_items(&mut self, mark: Mark) {
        let tree_id = item_tree::TreeId::new(self.expander.current_file_id(), None);
        let item_tree = tree_id.item_tree(self.db);
        let iter: SmallVec<[_; 2]> =
            item_tree.top_level_items().iter().filter_map(ModItem::as_assoc_item).collect();

        self.collect(&item_tree, tree_id, &iter);

        self.expander.exit(self.db, mark);
    }
}
