//! After we obtain a fresh AST fragment from a macro, code in this module helps to integrate
//! that fragment into the module structures that are already partially built.
//!
//! Items from the fragment are placed into modules,
//! unexpanded macros in the fragment are visited and registered.
//! Imports are also considered items and placed into modules here, but not resolved yet.

use std::cell::Cell;
use std::sync::Arc;

use rustc_ast::visit::{self, AssocCtxt, Visitor, WalkItemKind};
use rustc_ast::{
    self as ast, AssocItem, AssocItemKind, Block, ConstItem, Delegation, Fn, ForeignItem,
    ForeignItemKind, Impl, Item, ItemKind, MetaItemKind, NodeId, StaticItem, StmtKind, TyAlias,
};
use rustc_attr_parsing as attr;
use rustc_expand::base::ResolverExpand;
use rustc_expand::expand::AstFragment;
use rustc_hir::def::{self, *};
use rustc_hir::def_id::{CRATE_DEF_ID, DefId, LocalDefId};
use rustc_metadata::creader::LoadedMacro;
use rustc_middle::metadata::ModChild;
use rustc_middle::ty::Feed;
use rustc_middle::{bug, ty};
use rustc_span::hygiene::{ExpnId, LocalExpnId, MacroKind};
use rustc_span::{Ident, Span, Symbol, kw, sym};
use tracing::debug;

use crate::Namespace::{MacroNS, TypeNS, ValueNS};
use crate::def_collector::collect_definitions;
use crate::imports::{ImportData, ImportKind};
use crate::macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};
use crate::{
    BindingKey, Determinacy, ExternPreludeEntry, Finalize, MacroData, Module, ModuleKind,
    ModuleOrUniformRoot, NameBinding, NameBindingData, NameBindingKind, ParentScope, PathResult,
    ResolutionError, Resolver, ResolverArenas, Segment, ToNameBinding, Used, VisResolutionError,
    errors,
};

type Res = def::Res<NodeId>;

impl<'ra, Id: Into<DefId>> ToNameBinding<'ra>
    for (Module<'ra>, ty::Visibility<Id>, Span, LocalExpnId)
{
    fn to_name_binding(self, arenas: &'ra ResolverArenas<'ra>) -> NameBinding<'ra> {
        arenas.alloc_name_binding(NameBindingData {
            kind: NameBindingKind::Module(self.0),
            ambiguity: None,
            warn_ambiguity: false,
            vis: self.1.to_def_id(),
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'ra, Id: Into<DefId>> ToNameBinding<'ra> for (Res, ty::Visibility<Id>, Span, LocalExpnId) {
    fn to_name_binding(self, arenas: &'ra ResolverArenas<'ra>) -> NameBinding<'ra> {
        arenas.alloc_name_binding(NameBindingData {
            kind: NameBindingKind::Res(self.0),
            ambiguity: None,
            warn_ambiguity: false,
            vis: self.1.to_def_id(),
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    pub(crate) fn define<T>(&mut self, parent: Module<'ra>, ident: Ident, ns: Namespace, def: T)
    where
        T: ToNameBinding<'ra>,
    {
        let binding = def.to_name_binding(self.arenas);
        let key = self.new_disambiguated_key(ident, ns);
        if let Err(old_binding) = self.try_define(parent, key, binding, false) {
            self.report_conflict(parent, ident, ns, old_binding, binding);
        }
    }

    /// Walks up the tree of definitions starting at `def_id`,
    /// stopping at the first encountered module.
    /// Parent block modules for arbitrary def-ids are not recorded for the local crate,
    /// and are not preserved in metadata for foreign crates, so block modules are never
    /// returned by this function.
    ///
    /// For the local crate ignoring block modules may be incorrect, so use this method with care.
    ///
    /// For foreign crates block modules can be ignored without introducing observable differences,
    /// moreover they has to be ignored right now because they are not kept in metadata.
    /// Foreign parent modules are used for resolving names used by foreign macros with def-site
    /// hygiene, therefore block module ignorability relies on macros with def-site hygiene and
    /// block module parents being unreachable from other crates.
    /// Reachable macros with block module parents exist due to `#[macro_export] macro_rules!`,
    /// but they cannot use def-site hygiene, so the assumption holds
    /// (<https://github.com/rust-lang/rust/pull/77984#issuecomment-712445508>).
    pub(crate) fn get_nearest_non_block_module(&mut self, mut def_id: DefId) -> Module<'ra> {
        loop {
            match self.get_module(def_id) {
                Some(module) => return module,
                None => def_id = self.tcx.parent(def_id),
            }
        }
    }

    pub(crate) fn expect_module(&mut self, def_id: DefId) -> Module<'ra> {
        self.get_module(def_id).expect("argument `DefId` is not a module")
    }

    /// If `def_id` refers to a module (in resolver's sense, i.e. a module item, crate root, enum,
    /// or trait), then this function returns that module's resolver representation, otherwise it
    /// returns `None`.
    pub(crate) fn get_module(&mut self, def_id: DefId) -> Option<Module<'ra>> {
        if let module @ Some(..) = self.module_map.get(&def_id) {
            return module.copied();
        }

        if !def_id.is_local() {
            // Query `def_kind` is not used because query system overhead is too expensive here.
            let def_kind = self.cstore().def_kind_untracked(def_id);
            if let DefKind::Mod | DefKind::Enum | DefKind::Trait = def_kind {
                let parent = self
                    .tcx
                    .opt_parent(def_id)
                    .map(|parent_id| self.get_nearest_non_block_module(parent_id));
                // Query `expn_that_defined` is not used because
                // hashing spans in its result is expensive.
                let expn_id = self.cstore().expn_that_defined_untracked(def_id, self.tcx.sess);
                return Some(self.new_module(
                    parent,
                    ModuleKind::Def(def_kind, def_id, Some(self.tcx.item_name(def_id))),
                    expn_id,
                    self.def_span(def_id),
                    // FIXME: Account for `#[no_implicit_prelude]` attributes.
                    parent.is_some_and(|module| module.no_implicit_prelude),
                ));
            }
        }

        None
    }

    pub(crate) fn expn_def_scope(&mut self, expn_id: ExpnId) -> Module<'ra> {
        match expn_id.expn_data().macro_def_id {
            Some(def_id) => self.macro_def_scope(def_id),
            None => expn_id
                .as_local()
                .and_then(|expn_id| self.ast_transform_scopes.get(&expn_id).copied())
                .unwrap_or(self.graph_root),
        }
    }

    pub(crate) fn macro_def_scope(&mut self, def_id: DefId) -> Module<'ra> {
        if let Some(id) = def_id.as_local() {
            self.local_macro_def_scopes[&id]
        } else {
            self.get_nearest_non_block_module(def_id)
        }
    }

    pub(crate) fn get_macro(&mut self, res: Res) -> Option<&MacroData> {
        match res {
            Res::Def(DefKind::Macro(..), def_id) => Some(self.get_macro_by_def_id(def_id)),
            Res::NonMacroAttr(_) => Some(&self.non_macro_attr),
            _ => None,
        }
    }

    pub(crate) fn get_macro_by_def_id(&mut self, def_id: DefId) -> &MacroData {
        if self.macro_map.contains_key(&def_id) {
            return &self.macro_map[&def_id];
        }

        let loaded_macro = self.cstore().load_macro_untracked(def_id, self.tcx);
        let macro_data = match loaded_macro {
            LoadedMacro::MacroDef { def, ident, attrs, span, edition } => {
                self.compile_macro(&def, ident, &attrs, span, ast::DUMMY_NODE_ID, edition)
            }
            LoadedMacro::ProcMacro(ext) => MacroData::new(Arc::new(ext)),
        };

        self.macro_map.entry(def_id).or_insert(macro_data)
    }

    pub(crate) fn build_reduced_graph(
        &mut self,
        fragment: &AstFragment,
        parent_scope: ParentScope<'ra>,
    ) -> MacroRulesScopeRef<'ra> {
        collect_definitions(self, fragment, parent_scope.expansion);
        let mut visitor = BuildReducedGraphVisitor { r: self, parent_scope };
        fragment.visit_with(&mut visitor);
        visitor.parent_scope.macro_rules
    }

    pub(crate) fn build_reduced_graph_external(&mut self, module: Module<'ra>) {
        for child in self.tcx.module_children(module.def_id()) {
            let parent_scope = ParentScope::module(module, self);
            self.build_reduced_graph_for_external_crate_res(child, parent_scope)
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_res(
        &mut self,
        child: &ModChild,
        parent_scope: ParentScope<'ra>,
    ) {
        let parent = parent_scope.module;
        let ModChild { ident, res, vis, ref reexport_chain } = *child;
        let span = self.def_span(
            reexport_chain
                .first()
                .and_then(|reexport| reexport.id())
                .unwrap_or_else(|| res.def_id()),
        );
        let res = res.expect_non_local();
        let expansion = parent_scope.expansion;
        // Record primary definitions.
        match res {
            Res::Def(DefKind::Mod | DefKind::Enum | DefKind::Trait, def_id) => {
                let module = self.expect_module(def_id);
                self.define(parent, ident, TypeNS, (module, vis, span, expansion));
            }
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Variant
                | DefKind::TyAlias
                | DefKind::ForeignTy
                | DefKind::OpaqueTy
                | DefKind::TraitAlias
                | DefKind::AssocTy,
                _,
            )
            | Res::PrimTy(..)
            | Res::ToolMod => self.define(parent, ident, TypeNS, (res, vis, span, expansion)),
            Res::Def(
                DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Static { .. }
                | DefKind::Const
                | DefKind::AssocConst
                | DefKind::Ctor(..),
                _,
            ) => self.define(parent, ident, ValueNS, (res, vis, span, expansion)),
            Res::Def(DefKind::Macro(..), _) | Res::NonMacroAttr(..) => {
                self.define(parent, ident, MacroNS, (res, vis, span, expansion))
            }
            Res::Def(
                DefKind::TyParam
                | DefKind::ConstParam
                | DefKind::ExternCrate
                | DefKind::Use
                | DefKind::ForeignMod
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::Field
                | DefKind::LifetimeParam
                | DefKind::GlobalAsm
                | DefKind::Closure
                | DefKind::SyntheticCoroutineBody
                | DefKind::Impl { .. },
                _,
            )
            | Res::Local(..)
            | Res::SelfTyParam { .. }
            | Res::SelfTyAlias { .. }
            | Res::SelfCtor(..)
            | Res::Err => bug!("unexpected resolution: {:?}", res),
        }
    }
}

struct BuildReducedGraphVisitor<'a, 'ra, 'tcx> {
    r: &'a mut Resolver<'ra, 'tcx>,
    parent_scope: ParentScope<'ra>,
}

impl<'ra, 'tcx> AsMut<Resolver<'ra, 'tcx>> for BuildReducedGraphVisitor<'_, 'ra, 'tcx> {
    fn as_mut(&mut self) -> &mut Resolver<'ra, 'tcx> {
        self.r
    }
}

impl<'a, 'ra, 'tcx> BuildReducedGraphVisitor<'a, 'ra, 'tcx> {
    fn res(&self, def_id: impl Into<DefId>) -> Res {
        let def_id = def_id.into();
        Res::Def(self.r.tcx.def_kind(def_id), def_id)
    }

    fn resolve_visibility(&mut self, vis: &ast::Visibility) -> ty::Visibility {
        self.try_resolve_visibility(vis, true).unwrap_or_else(|err| {
            self.r.report_vis_error(err);
            ty::Visibility::Public
        })
    }

    fn try_resolve_visibility<'ast>(
        &mut self,
        vis: &'ast ast::Visibility,
        finalize: bool,
    ) -> Result<ty::Visibility, VisResolutionError<'ast>> {
        let parent_scope = &self.parent_scope;
        match vis.kind {
            ast::VisibilityKind::Public => Ok(ty::Visibility::Public),
            ast::VisibilityKind::Inherited => {
                Ok(match self.parent_scope.module.kind {
                    // Any inherited visibility resolved directly inside an enum or trait
                    // (i.e. variants, fields, and trait items) inherits from the visibility
                    // of the enum or trait.
                    ModuleKind::Def(DefKind::Enum | DefKind::Trait, def_id, _) => {
                        self.r.tcx.visibility(def_id).expect_local()
                    }
                    // Otherwise, the visibility is restricted to the nearest parent `mod` item.
                    _ => ty::Visibility::Restricted(
                        self.parent_scope.module.nearest_parent_mod().expect_local(),
                    ),
                })
            }
            ast::VisibilityKind::Restricted { ref path, id, .. } => {
                // For visibilities we are not ready to provide correct implementation of "uniform
                // paths" right now, so on 2018 edition we only allow module-relative paths for now.
                // On 2015 edition visibilities are resolved as crate-relative by default,
                // so we are prepending a root segment if necessary.
                let ident = path.segments.get(0).expect("empty path in visibility").ident;
                let crate_root = if ident.is_path_segment_keyword() {
                    None
                } else if ident.span.is_rust_2015() {
                    Some(Segment::from_ident(Ident::new(
                        kw::PathRoot,
                        path.span.shrink_to_lo().with_ctxt(ident.span.ctxt()),
                    )))
                } else {
                    return Err(VisResolutionError::Relative2018(ident.span, path));
                };

                let segments = crate_root
                    .into_iter()
                    .chain(path.segments.iter().map(|seg| seg.into()))
                    .collect::<Vec<_>>();
                let expected_found_error = |res| {
                    Err(VisResolutionError::ExpectedFound(
                        path.span,
                        Segment::names_to_string(&segments),
                        res,
                    ))
                };
                match self.r.resolve_path(
                    &segments,
                    None,
                    parent_scope,
                    finalize.then(|| Finalize::new(id, path.span)),
                    None,
                    None,
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                        let res = module.res().expect("visibility resolved to unnamed block");
                        if finalize {
                            self.r.record_partial_res(id, PartialRes::new(res));
                        }
                        if module.is_normal() {
                            match res {
                                Res::Err => Ok(ty::Visibility::Public),
                                _ => {
                                    let vis = ty::Visibility::Restricted(res.def_id());
                                    if self.r.is_accessible_from(vis, parent_scope.module) {
                                        Ok(vis.expect_local())
                                    } else {
                                        Err(VisResolutionError::AncestorOnly(path.span))
                                    }
                                }
                            }
                        } else {
                            expected_found_error(res)
                        }
                    }
                    PathResult::Module(..) => Err(VisResolutionError::ModuleOnly(path.span)),
                    PathResult::NonModule(partial_res) => {
                        expected_found_error(partial_res.expect_full_res())
                    }
                    PathResult::Failed { span, label, suggestion, .. } => {
                        Err(VisResolutionError::FailedToResolve(span, label, suggestion))
                    }
                    PathResult::Indeterminate => Err(VisResolutionError::Indeterminate(path.span)),
                }
            }
        }
    }

    fn insert_field_idents(&mut self, def_id: LocalDefId, fields: &[ast::FieldDef]) {
        if fields.iter().any(|field| field.is_placeholder) {
            // The fields are not expanded yet.
            return;
        }
        let fields = fields
            .iter()
            .enumerate()
            .map(|(i, field)| {
                field.ident.unwrap_or_else(|| Ident::from_str_and_span(&format!("{i}"), field.span))
            })
            .collect();
        self.r.field_names.insert(def_id, fields);
    }

    fn insert_field_visibilities_local(&mut self, def_id: DefId, fields: &[ast::FieldDef]) {
        let field_vis = fields
            .iter()
            .map(|field| field.vis.span.until(field.ident.map_or(field.ty.span, |i| i.span)))
            .collect();
        self.r.field_visibility_spans.insert(def_id, field_vis);
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        // If any statements are items, we need to create an anonymous module
        block
            .stmts
            .iter()
            .any(|statement| matches!(statement.kind, StmtKind::Item(_) | StmtKind::MacCall(_)))
    }

    // Add an import to the current module.
    fn add_import(
        &mut self,
        module_path: Vec<Segment>,
        kind: ImportKind<'ra>,
        span: Span,
        item: &ast::Item,
        root_span: Span,
        root_id: NodeId,
        vis: ty::Visibility,
    ) {
        let current_module = self.parent_scope.module;
        let import = self.r.arenas.alloc_import(ImportData {
            kind,
            parent_scope: self.parent_scope,
            module_path,
            imported_module: Cell::new(None),
            span,
            use_span: item.span,
            use_span_with_attributes: item.span_with_attributes(),
            has_attributes: !item.attrs.is_empty(),
            root_span,
            root_id,
            vis,
        });

        self.r.indeterminate_imports.push(import);
        match import.kind {
            // Don't add unresolved underscore imports to modules
            ImportKind::Single { target: Ident { name: kw::Underscore, .. }, .. } => {}
            ImportKind::Single { target, type_ns_only, .. } => {
                self.r.per_ns(|this, ns| {
                    if !type_ns_only || ns == TypeNS {
                        let key = BindingKey::new(target, ns);
                        let mut resolution = this.resolution(current_module, key).borrow_mut();
                        resolution.single_imports.insert(import);
                    }
                });
            }
            // We don't add prelude imports to the globs since they only affect lexical scopes,
            // which are not relevant to import resolution.
            ImportKind::Glob { is_prelude: true, .. } => {}
            ImportKind::Glob { .. } => current_module.globs.borrow_mut().push(import),
            _ => unreachable!(),
        }
    }

    fn build_reduced_graph_for_use_tree(
        &mut self,
        // This particular use tree
        use_tree: &ast::UseTree,
        id: NodeId,
        parent_prefix: &[Segment],
        nested: bool,
        list_stem: bool,
        // The whole `use` item
        item: &Item,
        vis: ty::Visibility,
        root_span: Span,
    ) {
        debug!(
            "build_reduced_graph_for_use_tree(parent_prefix={:?}, use_tree={:?}, nested={})",
            parent_prefix, use_tree, nested
        );

        // Top level use tree reuses the item's id and list stems reuse their parent
        // use tree's ids, so in both cases their visibilities are already filled.
        if nested && !list_stem {
            self.r.feed_visibility(self.r.feed(id), vis);
        }

        let mut prefix_iter = parent_prefix
            .iter()
            .cloned()
            .chain(use_tree.prefix.segments.iter().map(|seg| seg.into()))
            .peekable();

        // On 2015 edition imports are resolved as crate-relative by default,
        // so prefixes are prepended with crate root segment if necessary.
        // The root is prepended lazily, when the first non-empty prefix or terminating glob
        // appears, so imports in braced groups can have roots prepended independently.
        let is_glob = matches!(use_tree.kind, ast::UseTreeKind::Glob);
        let crate_root = match prefix_iter.peek() {
            Some(seg) if !seg.ident.is_path_segment_keyword() && seg.ident.span.is_rust_2015() => {
                Some(seg.ident.span.ctxt())
            }
            None if is_glob && use_tree.span.is_rust_2015() => Some(use_tree.span.ctxt()),
            _ => None,
        }
        .map(|ctxt| {
            Segment::from_ident(Ident::new(
                kw::PathRoot,
                use_tree.prefix.span.shrink_to_lo().with_ctxt(ctxt),
            ))
        });

        let prefix = crate_root.into_iter().chain(prefix_iter).collect::<Vec<_>>();
        debug!("build_reduced_graph_for_use_tree: prefix={:?}", prefix);

        let empty_for_self = |prefix: &[Segment]| {
            prefix.is_empty() || prefix.len() == 1 && prefix[0].ident.name == kw::PathRoot
        };
        match use_tree.kind {
            ast::UseTreeKind::Simple(rename) => {
                let mut ident = use_tree.ident();
                let mut module_path = prefix;
                let mut source = module_path.pop().unwrap();
                let mut type_ns_only = false;

                if nested {
                    // Correctly handle `self`
                    if source.ident.name == kw::SelfLower {
                        type_ns_only = true;

                        if empty_for_self(&module_path) {
                            self.r.report_error(
                                use_tree.span,
                                ResolutionError::SelfImportOnlyInImportListWithNonEmptyPrefix,
                            );
                            return;
                        }

                        // Replace `use foo::{ self };` with `use foo;`
                        let self_span = source.ident.span;
                        source = module_path.pop().unwrap();
                        if rename.is_none() {
                            // Keep the span of `self`, but the name of `foo`
                            ident = Ident { name: source.ident.name, span: self_span };
                        }
                    }
                } else {
                    // Disallow `self`
                    if source.ident.name == kw::SelfLower {
                        let parent = module_path.last();

                        let span = match parent {
                            // only `::self` from `use foo::self as bar`
                            Some(seg) => seg.ident.span.shrink_to_hi().to(source.ident.span),
                            None => source.ident.span,
                        };
                        let span_with_rename = match rename {
                            // only `self as bar` from `use foo::self as bar`
                            Some(rename) => source.ident.span.to(rename.span),
                            None => source.ident.span,
                        };
                        self.r.report_error(
                            span,
                            ResolutionError::SelfImportsOnlyAllowedWithin {
                                root: parent.is_none(),
                                span_with_rename,
                            },
                        );

                        // Error recovery: replace `use foo::self;` with `use foo;`
                        if let Some(parent) = module_path.pop() {
                            source = parent;
                            if rename.is_none() {
                                ident = source.ident;
                            }
                        }
                    }

                    // Disallow `use $crate;`
                    if source.ident.name == kw::DollarCrate && module_path.is_empty() {
                        let crate_root = self.r.resolve_crate_root(source.ident);
                        let crate_name = match crate_root.kind {
                            ModuleKind::Def(.., name) => name,
                            ModuleKind::Block => unreachable!(),
                        };
                        // HACK(eddyb) unclear how good this is, but keeping `$crate`
                        // in `source` breaks `tests/ui/imports/import-crate-var.rs`,
                        // while the current crate doesn't have a valid `crate_name`.
                        if let Some(crate_name) = crate_name {
                            // `crate_name` should not be interpreted as relative.
                            module_path.push(Segment::from_ident_and_id(
                                Ident { name: kw::PathRoot, span: source.ident.span },
                                self.r.next_node_id(),
                            ));
                            source.ident.name = crate_name;
                        }
                        if rename.is_none() {
                            ident.name = sym::dummy;
                        }

                        self.r.dcx().emit_err(errors::CrateImported { span: item.span });
                    }
                }

                if ident.name == kw::Crate {
                    self.r.dcx().emit_err(errors::UnnamedCrateRootImport { span: ident.span });
                }

                let kind = ImportKind::Single {
                    source: source.ident,
                    target: ident,
                    source_bindings: PerNS {
                        type_ns: Cell::new(Err(Determinacy::Undetermined)),
                        value_ns: Cell::new(Err(Determinacy::Undetermined)),
                        macro_ns: Cell::new(Err(Determinacy::Undetermined)),
                    },
                    target_bindings: PerNS {
                        type_ns: Cell::new(None),
                        value_ns: Cell::new(None),
                        macro_ns: Cell::new(None),
                    },
                    type_ns_only,
                    nested,
                    id,
                };

                self.add_import(module_path, kind, use_tree.span, item, root_span, item.id, vis);
            }
            ast::UseTreeKind::Glob => {
                let kind = ImportKind::Glob {
                    is_prelude: ast::attr::contains_name(&item.attrs, sym::prelude_import),
                    max_vis: Cell::new(None),
                    id,
                };

                self.add_import(prefix, kind, use_tree.span, item, root_span, item.id, vis);
            }
            ast::UseTreeKind::Nested { ref items, .. } => {
                // Ensure there is at most one `self` in the list
                let self_spans = items
                    .iter()
                    .filter_map(|(use_tree, _)| {
                        if let ast::UseTreeKind::Simple(..) = use_tree.kind
                            && use_tree.ident().name == kw::SelfLower
                        {
                            return Some(use_tree.span);
                        }

                        None
                    })
                    .collect::<Vec<_>>();
                if self_spans.len() > 1 {
                    let mut e = self.r.into_struct_error(
                        self_spans[0],
                        ResolutionError::SelfImportCanOnlyAppearOnceInTheList,
                    );

                    for other_span in self_spans.iter().skip(1) {
                        e.span_label(*other_span, "another `self` import appears here");
                    }

                    e.emit();
                }

                for &(ref tree, id) in items {
                    self.build_reduced_graph_for_use_tree(
                        // This particular use tree
                        tree, id, &prefix, true, false, // The whole `use` item
                        item, vis, root_span,
                    );
                }

                // Empty groups `a::b::{}` are turned into synthetic `self` imports
                // `a::b::c::{self as _}`, so that their prefixes are correctly
                // resolved and checked for privacy/stability/etc.
                if items.is_empty() && !empty_for_self(&prefix) {
                    let new_span = prefix[prefix.len() - 1].ident.span;
                    let tree = ast::UseTree {
                        prefix: ast::Path::from_ident(Ident::new(kw::SelfLower, new_span)),
                        kind: ast::UseTreeKind::Simple(Some(Ident::new(kw::Underscore, new_span))),
                        span: use_tree.span,
                    };
                    self.build_reduced_graph_for_use_tree(
                        // This particular use tree
                        &tree,
                        id,
                        &prefix,
                        true,
                        true,
                        // The whole `use` item
                        item,
                        ty::Visibility::Restricted(
                            self.parent_scope.module.nearest_parent_mod().expect_local(),
                        ),
                        root_span,
                    );
                }
            }
        }
    }

    fn build_reduced_graph_for_struct_variant(
        &mut self,
        fields: &[ast::FieldDef],
        ident: Ident,
        feed: Feed<'tcx, LocalDefId>,
        adt_res: Res,
        adt_vis: ty::Visibility,
        adt_span: Span,
    ) {
        let parent_scope = &self.parent_scope;
        let parent = parent_scope.module;
        let expansion = parent_scope.expansion;

        // Define a name in the type namespace if it is not anonymous.
        self.r.define(parent, ident, TypeNS, (adt_res, adt_vis, adt_span, expansion));
        self.r.feed_visibility(feed, adt_vis);
        let def_id = feed.key();

        // Record field names for error reporting.
        self.insert_field_idents(def_id, fields);
        self.insert_field_visibilities_local(def_id.to_def_id(), fields);
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &'a Item) {
        let parent_scope = &self.parent_scope;
        let parent = parent_scope.module;
        let expansion = parent_scope.expansion;
        let sp = item.span;
        let vis = self.resolve_visibility(&item.vis);
        let feed = self.r.feed(item.id);
        let local_def_id = feed.key();
        let def_id = local_def_id.to_def_id();
        let def_kind = self.r.tcx.def_kind(def_id);
        let res = Res::Def(def_kind, def_id);

        self.r.feed_visibility(feed, vis);

        match item.kind {
            ItemKind::Use(ref use_tree) => {
                self.build_reduced_graph_for_use_tree(
                    // This particular use tree
                    use_tree,
                    item.id,
                    &[],
                    false,
                    false,
                    // The whole `use` item
                    item,
                    vis,
                    use_tree.span,
                );
            }

            ItemKind::ExternCrate(orig_name, ident) => {
                self.build_reduced_graph_for_extern_crate(
                    orig_name,
                    item,
                    ident,
                    local_def_id,
                    vis,
                    parent,
                );
            }

            ItemKind::Mod(_, ident, ref mod_kind) => {
                let module = self.r.new_module(
                    Some(parent),
                    ModuleKind::Def(def_kind, def_id, Some(ident.name)),
                    expansion.to_expn_id(),
                    item.span,
                    parent.no_implicit_prelude
                        || ast::attr::contains_name(&item.attrs, sym::no_implicit_prelude),
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));

                if let ast::ModKind::Loaded(_, _, _, Err(_)) = mod_kind {
                    self.r.mods_with_parse_errors.insert(def_id);
                }

                // Descend into the module.
                self.parent_scope.module = module;
            }

            // These items live in the value namespace.
            ItemKind::Const(box ConstItem { ident, .. })
            | ItemKind::Delegation(box Delegation { ident, .. })
            | ItemKind::Static(box StaticItem { ident, .. }) => {
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));
            }
            ItemKind::Fn(box Fn { ident, .. }) => {
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));

                // Functions introducing procedural macros reserve a slot
                // in the macro namespace as well (see #52225).
                self.define_macro(item);
            }

            // These items live in the type namespace.
            ItemKind::TyAlias(box TyAlias { ident, .. }) | ItemKind::TraitAlias(ident, ..) => {
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));
            }

            ItemKind::Enum(ident, _, _) | ItemKind::Trait(box ast::Trait { ident, .. }) => {
                let module = self.r.new_module(
                    Some(parent),
                    ModuleKind::Def(def_kind, def_id, Some(ident.name)),
                    expansion.to_expn_id(),
                    item.span,
                    parent.no_implicit_prelude,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.parent_scope.module = module;
            }

            // These items live in both the type and value namespaces.
            ItemKind::Struct(ident, ref vdata, _) => {
                self.build_reduced_graph_for_struct_variant(
                    vdata.fields(),
                    ident,
                    feed,
                    res,
                    vis,
                    sp,
                );

                // If this is a tuple or unit struct, define a name
                // in the value namespace as well.
                if let Some(ctor_node_id) = vdata.ctor_node_id() {
                    // If the structure is marked as non_exhaustive then lower the visibility
                    // to within the crate.
                    let mut ctor_vis = if vis.is_public()
                        && ast::attr::contains_name(&item.attrs, sym::non_exhaustive)
                    {
                        ty::Visibility::Restricted(CRATE_DEF_ID)
                    } else {
                        vis
                    };

                    let mut ret_fields = Vec::with_capacity(vdata.fields().len());

                    for field in vdata.fields() {
                        // NOTE: The field may be an expansion placeholder, but expansion sets
                        // correct visibilities for unnamed field placeholders specifically, so the
                        // constructor visibility should still be determined correctly.
                        let field_vis = self
                            .try_resolve_visibility(&field.vis, false)
                            .unwrap_or(ty::Visibility::Public);
                        if ctor_vis.is_at_least(field_vis, self.r.tcx) {
                            ctor_vis = field_vis;
                        }
                        ret_fields.push(field_vis.to_def_id());
                    }
                    let feed = self.r.feed(ctor_node_id);
                    let ctor_def_id = feed.key();
                    let ctor_res = self.res(ctor_def_id);
                    self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, sp, expansion));
                    self.r.feed_visibility(feed, ctor_vis);
                    // We need the field visibility spans also for the constructor for E0603.
                    self.insert_field_visibilities_local(ctor_def_id.to_def_id(), vdata.fields());

                    self.r
                        .struct_constructors
                        .insert(local_def_id, (ctor_res, ctor_vis.to_def_id(), ret_fields));
                }
            }

            ItemKind::Union(ident, ref vdata, _) => {
                self.build_reduced_graph_for_struct_variant(
                    vdata.fields(),
                    ident,
                    feed,
                    res,
                    vis,
                    sp,
                );
            }

            // These items do not add names to modules.
            ItemKind::Impl(box Impl { of_trait: Some(..), .. })
            | ItemKind::Impl { .. }
            | ItemKind::ForeignMod(..)
            | ItemKind::GlobalAsm(..) => {}

            ItemKind::MacroDef(..) | ItemKind::MacCall(_) | ItemKind::DelegationMac(..) => {
                unreachable!()
            }
        }
    }

    fn build_reduced_graph_for_extern_crate(
        &mut self,
        orig_name: Option<Symbol>,
        item: &Item,
        ident: Ident,
        local_def_id: LocalDefId,
        vis: ty::Visibility,
        parent: Module<'ra>,
    ) {
        let sp = item.span;
        let parent_scope = self.parent_scope;
        let expansion = parent_scope.expansion;

        let (used, module, binding) = if orig_name.is_none() && ident.name == kw::SelfLower {
            self.r.dcx().emit_err(errors::ExternCrateSelfRequiresRenaming { span: sp });
            return;
        } else if orig_name == Some(kw::SelfLower) {
            Some(self.r.graph_root)
        } else {
            let tcx = self.r.tcx;
            let crate_id = self.r.crate_loader(|c| {
                c.process_extern_crate(item, local_def_id, &tcx.definitions_untracked())
            });
            crate_id.map(|crate_id| {
                self.r.extern_crate_map.insert(local_def_id, crate_id);
                self.r.expect_module(crate_id.as_def_id())
            })
        }
        .map(|module| {
            let used = self.process_macro_use_imports(item, module);
            let vis = ty::Visibility::<LocalDefId>::Public;
            let binding = (module, vis, sp, expansion).to_name_binding(self.r.arenas);
            (used, Some(ModuleOrUniformRoot::Module(module)), binding)
        })
        .unwrap_or((true, None, self.r.dummy_binding));
        let import = self.r.arenas.alloc_import(ImportData {
            kind: ImportKind::ExternCrate { source: orig_name, target: ident, id: item.id },
            root_id: item.id,
            parent_scope: self.parent_scope,
            imported_module: Cell::new(module),
            has_attributes: !item.attrs.is_empty(),
            use_span_with_attributes: item.span_with_attributes(),
            use_span: item.span,
            root_span: item.span,
            span: item.span,
            module_path: Vec::new(),
            vis,
        });
        if used {
            self.r.import_use_map.insert(import, Used::Other);
        }
        self.r.potentially_unused_imports.push(import);
        let imported_binding = self.r.import(binding, import);
        if parent == self.r.graph_root {
            let ident = ident.normalize_to_macros_2_0();
            if let Some(entry) = self.r.extern_prelude.get(&ident)
                && expansion != LocalExpnId::ROOT
                && orig_name.is_some()
                && !entry.is_import()
            {
                self.r.dcx().emit_err(
                    errors::MacroExpandedExternCrateCannotShadowExternArguments { span: item.span },
                );
                // `return` is intended to discard this binding because it's an
                // unregistered ambiguity error which would result in a panic
                // caused by inconsistency `path_res`
                // more details: https://github.com/rust-lang/rust/pull/111761
                return;
            }
            let entry = self
                .r
                .extern_prelude
                .entry(ident)
                .or_insert(ExternPreludeEntry { binding: None, introduced_by_item: true });
            if orig_name.is_some() {
                entry.introduced_by_item = true;
            }
            // Binding from `extern crate` item in source code can replace
            // a binding from `--extern` on command line here.
            if !entry.is_import() {
                entry.binding = Some(imported_binding)
            } else if ident.name != kw::Underscore {
                self.r.dcx().span_delayed_bug(
                    item.span,
                    format!("it had been define the external module '{ident}' multiple times"),
                );
            }
        }
        self.r.define(parent, ident, TypeNS, imported_binding);
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self, item: &ForeignItem, ident: Ident) {
        let feed = self.r.feed(item.id);
        let local_def_id = feed.key();
        let def_id = local_def_id.to_def_id();
        let ns = match item.kind {
            ForeignItemKind::Fn(..) => ValueNS,
            ForeignItemKind::Static(..) => ValueNS,
            ForeignItemKind::TyAlias(..) => TypeNS,
            ForeignItemKind::MacCall(..) => unreachable!(),
        };
        let parent = self.parent_scope.module;
        let expansion = self.parent_scope.expansion;
        let vis = self.resolve_visibility(&item.vis);
        self.r.define(parent, ident, ns, (self.res(def_id), vis, item.span, expansion));
        self.r.feed_visibility(feed, vis);
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block) {
        let parent = self.parent_scope.module;
        let expansion = self.parent_scope.expansion;
        if self.block_needs_anonymous_module(block) {
            let module = self.r.new_module(
                Some(parent),
                ModuleKind::Block,
                expansion.to_expn_id(),
                block.span,
                parent.no_implicit_prelude,
            );
            self.r.block_map.insert(block.id, module);
            self.parent_scope.module = module; // Descend into the block.
        }
    }

    fn add_macro_use_binding(
        &mut self,
        name: Symbol,
        binding: NameBinding<'ra>,
        span: Span,
        allow_shadowing: bool,
    ) {
        if self.r.macro_use_prelude.insert(name, binding).is_some() && !allow_shadowing {
            self.r.dcx().emit_err(errors::MacroUseNameAlreadyInUse { span, name });
        }
    }

    /// Returns `true` if we should consider the underlying `extern crate` to be used.
    fn process_macro_use_imports(&mut self, item: &Item, module: Module<'ra>) -> bool {
        let mut import_all = None;
        let mut single_imports = Vec::new();
        for attr in &item.attrs {
            if attr.has_name(sym::macro_use) {
                if self.parent_scope.module.parent.is_some() {
                    self.r.dcx().emit_err(errors::ExternCrateLoadingMacroNotAtCrateRoot {
                        span: item.span,
                    });
                }
                if let ItemKind::ExternCrate(Some(orig_name), _) = item.kind
                    && orig_name == kw::SelfLower
                {
                    self.r.dcx().emit_err(errors::MacroUseExternCrateSelf { span: attr.span });
                }
                let ill_formed = |span| {
                    self.r.dcx().emit_err(errors::BadMacroImport { span });
                };
                match attr.meta() {
                    Some(meta) => match meta.kind {
                        MetaItemKind::Word => {
                            import_all = Some(meta.span);
                            break;
                        }
                        MetaItemKind::List(meta_item_inners) => {
                            for meta_item_inner in meta_item_inners {
                                match meta_item_inner.ident() {
                                    Some(ident) if meta_item_inner.is_word() => {
                                        single_imports.push(ident)
                                    }
                                    _ => ill_formed(meta_item_inner.span()),
                                }
                            }
                        }
                        MetaItemKind::NameValue(..) => ill_formed(meta.span),
                    },
                    None => ill_formed(attr.span),
                }
            }
        }

        let macro_use_import = |this: &Self, span, warn_private| {
            this.r.arenas.alloc_import(ImportData {
                kind: ImportKind::MacroUse { warn_private },
                root_id: item.id,
                parent_scope: this.parent_scope,
                imported_module: Cell::new(Some(ModuleOrUniformRoot::Module(module))),
                use_span_with_attributes: item.span_with_attributes(),
                has_attributes: !item.attrs.is_empty(),
                use_span: item.span,
                root_span: span,
                span,
                module_path: Vec::new(),
                vis: ty::Visibility::Restricted(CRATE_DEF_ID),
            })
        };

        let allow_shadowing = self.parent_scope.expansion == LocalExpnId::ROOT;
        if let Some(span) = import_all {
            let import = macro_use_import(self, span, false);
            self.r.potentially_unused_imports.push(import);
            module.for_each_child(self, |this, ident, ns, binding| {
                if ns == MacroNS {
                    let imported_binding =
                        if this.r.is_accessible_from(binding.vis, this.parent_scope.module) {
                            this.r.import(binding, import)
                        } else if !this.r.is_builtin_macro(binding.res())
                            && !this.r.macro_use_prelude.contains_key(&ident.name)
                        {
                            // - `!r.is_builtin_macro(res)` excluding the built-in macros such as `Debug` or `Hash`.
                            // - `!r.macro_use_prelude.contains_key(name)` excluding macros defined in other extern
                            //    crates such as `std`.
                            // FIXME: This branch should eventually be removed.
                            let import = macro_use_import(this, span, true);
                            this.r.import(binding, import)
                        } else {
                            return;
                        };
                    this.add_macro_use_binding(ident.name, imported_binding, span, allow_shadowing);
                }
            });
        } else {
            for ident in single_imports.iter().cloned() {
                let result = self.r.maybe_resolve_ident_in_module(
                    ModuleOrUniformRoot::Module(module),
                    ident,
                    MacroNS,
                    &self.parent_scope,
                    None,
                );
                if let Ok(binding) = result {
                    let import = macro_use_import(self, ident.span, false);
                    self.r.potentially_unused_imports.push(import);
                    let imported_binding = self.r.import(binding, import);
                    self.add_macro_use_binding(
                        ident.name,
                        imported_binding,
                        ident.span,
                        allow_shadowing,
                    );
                } else {
                    self.r.dcx().emit_err(errors::ImportedMacroNotFound { span: ident.span });
                }
            }
        }
        import_all.is_some() || !single_imports.is_empty()
    }

    /// Returns `true` if this attribute list contains `macro_use`.
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.has_name(sym::macro_escape) {
                let inner_attribute = matches!(attr.style, ast::AttrStyle::Inner);
                self.r
                    .dcx()
                    .emit_warn(errors::MacroExternDeprecated { span: attr.span, inner_attribute });
            } else if !attr.has_name(sym::macro_use) {
                continue;
            }

            if !attr.is_word() {
                self.r.dcx().emit_err(errors::ArgumentsMacroUseNotAllowed { span: attr.span });
            }
            return true;
        }

        false
    }

    fn visit_invoc(&mut self, id: NodeId) -> LocalExpnId {
        let invoc_id = id.placeholder_to_expn_id();
        let old_parent_scope = self.r.invocation_parent_scopes.insert(invoc_id, self.parent_scope);
        assert!(old_parent_scope.is_none(), "invocation data is reset for an invocation");
        invoc_id
    }

    /// Visit invocation in context in which it can emit a named item (possibly `macro_rules`)
    /// directly into its parent scope's module.
    fn visit_invoc_in_module(&mut self, id: NodeId) -> MacroRulesScopeRef<'ra> {
        let invoc_id = self.visit_invoc(id);
        self.parent_scope.module.unexpanded_invocations.borrow_mut().insert(invoc_id);
        self.r.arenas.alloc_macro_rules_scope(MacroRulesScope::Invocation(invoc_id))
    }

    fn proc_macro_stub(
        &self,
        item: &ast::Item,
        fn_ident: Ident,
    ) -> Option<(MacroKind, Ident, Span)> {
        if ast::attr::contains_name(&item.attrs, sym::proc_macro) {
            return Some((MacroKind::Bang, fn_ident, item.span));
        } else if ast::attr::contains_name(&item.attrs, sym::proc_macro_attribute) {
            return Some((MacroKind::Attr, fn_ident, item.span));
        } else if let Some(attr) = ast::attr::find_by_name(&item.attrs, sym::proc_macro_derive)
            && let Some(meta_item_inner) =
                attr.meta_item_list().and_then(|list| list.get(0).cloned())
            && let Some(ident) = meta_item_inner.ident()
        {
            return Some((MacroKind::Derive, ident, ident.span));
        }
        None
    }

    // Mark the given macro as unused unless its name starts with `_`.
    // Macro uses will remove items from this set, and the remaining
    // items will be reported as `unused_macros`.
    fn insert_unused_macro(&mut self, ident: Ident, def_id: LocalDefId, node_id: NodeId) {
        if !ident.as_str().starts_with('_') {
            self.r.unused_macros.insert(def_id, (node_id, ident));
            for (rule_i, rule_span) in &self.r.macro_map[&def_id.to_def_id()].rule_spans {
                self.r
                    .unused_macro_rules
                    .entry(node_id)
                    .or_default()
                    .insert(*rule_i, (ident, *rule_span));
            }
        }
    }

    fn define_macro(&mut self, item: &ast::Item) -> MacroRulesScopeRef<'ra> {
        let parent_scope = self.parent_scope;
        let expansion = parent_scope.expansion;
        let feed = self.r.feed(item.id);
        let def_id = feed.key();
        let (res, ident, span, macro_rules) = match &item.kind {
            ItemKind::MacroDef(ident, def) => {
                (self.res(def_id), *ident, item.span, def.macro_rules)
            }
            ItemKind::Fn(box ast::Fn { ident: fn_ident, .. }) => {
                match self.proc_macro_stub(item, *fn_ident) {
                    Some((macro_kind, ident, span)) => {
                        let res = Res::Def(DefKind::Macro(macro_kind), def_id.to_def_id());
                        let macro_data = MacroData::new(self.r.dummy_ext(macro_kind));
                        self.r.macro_map.insert(def_id.to_def_id(), macro_data);
                        self.r.proc_macro_stubs.insert(def_id);
                        (res, ident, span, false)
                    }
                    None => return parent_scope.macro_rules,
                }
            }
            _ => unreachable!(),
        };

        self.r.local_macro_def_scopes.insert(def_id, parent_scope.module);

        if macro_rules {
            let ident = ident.normalize_to_macros_2_0();
            self.r.macro_names.insert(ident);
            let is_macro_export = ast::attr::contains_name(&item.attrs, sym::macro_export);
            let vis = if is_macro_export {
                ty::Visibility::Public
            } else {
                ty::Visibility::Restricted(CRATE_DEF_ID)
            };
            let binding = (res, vis, span, expansion).to_name_binding(self.r.arenas);
            self.r.set_binding_parent_module(binding, parent_scope.module);
            self.r.all_macro_rules.insert(ident.name);
            if is_macro_export {
                let import = self.r.arenas.alloc_import(ImportData {
                    kind: ImportKind::MacroExport,
                    root_id: item.id,
                    parent_scope: self.parent_scope,
                    imported_module: Cell::new(None),
                    has_attributes: false,
                    use_span_with_attributes: span,
                    use_span: span,
                    root_span: span,
                    span,
                    module_path: Vec::new(),
                    vis,
                });
                self.r.import_use_map.insert(import, Used::Other);
                let import_binding = self.r.import(binding, import);
                self.r.define(self.r.graph_root, ident, MacroNS, import_binding);
            } else {
                self.r.check_reserved_macro_name(ident, res);
                self.insert_unused_macro(ident, def_id, item.id);
            }
            self.r.feed_visibility(feed, vis);
            let scope = self.r.arenas.alloc_macro_rules_scope(MacroRulesScope::Binding(
                self.r.arenas.alloc_macro_rules_binding(MacroRulesBinding {
                    parent_macro_rules_scope: parent_scope.macro_rules,
                    binding,
                    ident,
                }),
            ));
            self.r.macro_rules_scopes.insert(def_id, scope);
            scope
        } else {
            let module = parent_scope.module;
            let vis = match item.kind {
                // Visibilities must not be resolved non-speculatively twice
                // and we already resolved this one as a `fn` item visibility.
                ItemKind::Fn(..) => {
                    self.try_resolve_visibility(&item.vis, false).unwrap_or(ty::Visibility::Public)
                }
                _ => self.resolve_visibility(&item.vis),
            };
            if !vis.is_public() {
                self.insert_unused_macro(ident, def_id, item.id);
            }
            self.r.define(module, ident, MacroNS, (res, vis, span, expansion));
            self.r.feed_visibility(feed, vis);
            self.parent_scope.macro_rules
        }
    }
}

macro_rules! method {
    ($visit:ident: $ty:ty, $invoc:path, $walk:ident) => {
        fn $visit(&mut self, node: &'a $ty) {
            if let $invoc(..) = node.kind {
                self.visit_invoc(node.id);
            } else {
                visit::$walk(self, node);
            }
        }
    };
}

impl<'a, 'ra, 'tcx> Visitor<'a> for BuildReducedGraphVisitor<'a, 'ra, 'tcx> {
    method!(visit_expr: ast::Expr, ast::ExprKind::MacCall, walk_expr);
    method!(visit_pat: ast::Pat, ast::PatKind::MacCall, walk_pat);
    method!(visit_ty: ast::Ty, ast::TyKind::MacCall, walk_ty);

    fn visit_item(&mut self, item: &'a Item) {
        let orig_module_scope = self.parent_scope.module;
        self.parent_scope.macro_rules = match item.kind {
            ItemKind::MacroDef(..) => {
                let macro_rules_scope = self.define_macro(item);
                visit::walk_item(self, item);
                macro_rules_scope
            }
            ItemKind::MacCall(..) => self.visit_invoc_in_module(item.id),
            _ => {
                let orig_macro_rules_scope = self.parent_scope.macro_rules;
                self.build_reduced_graph_for_item(item);
                match item.kind {
                    ItemKind::Mod(..) => {
                        // Visit attributes after items for backward compatibility.
                        // This way they can use `macro_rules` defined later.
                        self.visit_vis(&item.vis);
                        item.kind.walk(item.span, item.id, &item.vis, (), self);
                        visit::walk_list!(self, visit_attribute, &item.attrs);
                    }
                    _ => visit::walk_item(self, item),
                }
                match item.kind {
                    ItemKind::Mod(..) if self.contains_macro_use(&item.attrs) => {
                        self.parent_scope.macro_rules
                    }
                    _ => orig_macro_rules_scope,
                }
            }
        };
        self.parent_scope.module = orig_module_scope;
    }

    fn visit_stmt(&mut self, stmt: &'a ast::Stmt) {
        if let ast::StmtKind::MacCall(..) = stmt.kind {
            self.parent_scope.macro_rules = self.visit_invoc_in_module(stmt.id);
        } else {
            visit::walk_stmt(self, stmt);
        }
    }

    fn visit_foreign_item(&mut self, foreign_item: &'a ForeignItem) {
        let ident = match foreign_item.kind {
            ForeignItemKind::Static(box StaticItem { ident, .. })
            | ForeignItemKind::Fn(box Fn { ident, .. })
            | ForeignItemKind::TyAlias(box TyAlias { ident, .. }) => ident,
            ForeignItemKind::MacCall(_) => {
                self.visit_invoc_in_module(foreign_item.id);
                return;
            }
        };

        self.build_reduced_graph_for_foreign_item(foreign_item, ident);
        visit::walk_item(self, foreign_item);
    }

    fn visit_block(&mut self, block: &'a Block) {
        let orig_current_module = self.parent_scope.module;
        let orig_current_macro_rules_scope = self.parent_scope.macro_rules;
        self.build_reduced_graph_for_block(block);
        visit::walk_block(self, block);
        self.parent_scope.module = orig_current_module;
        self.parent_scope.macro_rules = orig_current_macro_rules_scope;
    }

    fn visit_assoc_item(&mut self, item: &'a AssocItem, ctxt: AssocCtxt) {
        let (ident, ns) = match item.kind {
            AssocItemKind::Const(box ConstItem { ident, .. })
            | AssocItemKind::Fn(box Fn { ident, .. })
            | AssocItemKind::Delegation(box Delegation { ident, .. }) => (ident, ValueNS),

            AssocItemKind::Type(box TyAlias { ident, .. }) => (ident, TypeNS),

            AssocItemKind::MacCall(_) => {
                match ctxt {
                    AssocCtxt::Trait => {
                        self.visit_invoc_in_module(item.id);
                    }
                    AssocCtxt::Impl { .. } => {
                        let invoc_id = item.id.placeholder_to_expn_id();
                        if !self.r.glob_delegation_invoc_ids.contains(&invoc_id) {
                            self.r
                                .impl_unexpanded_invocations
                                .entry(self.r.invocation_parent(invoc_id))
                                .or_default()
                                .insert(invoc_id);
                        }
                        self.visit_invoc(item.id);
                    }
                }
                return;
            }

            AssocItemKind::DelegationMac(..) => bug!(),
        };
        let vis = self.resolve_visibility(&item.vis);
        let feed = self.r.feed(item.id);
        let local_def_id = feed.key();
        let def_id = local_def_id.to_def_id();

        if !(matches!(ctxt, AssocCtxt::Impl { of_trait: true })
            && matches!(item.vis.kind, ast::VisibilityKind::Inherited))
        {
            // Trait impl item visibility is inherited from its trait when not specified
            // explicitly. In that case we cannot determine it here in early resolve,
            // so we leave a hole in the visibility table to be filled later.
            self.r.feed_visibility(feed, vis);
        }

        if ctxt == AssocCtxt::Trait {
            let parent = self.parent_scope.module;
            let expansion = self.parent_scope.expansion;
            self.r.define(parent, ident, ns, (self.res(def_id), vis, item.span, expansion));
        } else if !matches!(&item.kind, AssocItemKind::Delegation(deleg) if deleg.from_glob) {
            let impl_def_id = self.r.tcx.local_parent(local_def_id);
            let key = BindingKey::new(ident.normalize_to_macros_2_0(), ns);
            self.r.impl_binding_keys.entry(impl_def_id).or_default().insert(key);
        }

        visit::walk_assoc_item(self, item, ctxt);
    }

    fn visit_attribute(&mut self, attr: &'a ast::Attribute) {
        if !attr.is_doc_comment() && attr::is_builtin_attr(attr) {
            self.r
                .builtin_attrs
                .push((attr.get_normal_item().path.segments[0].ident, self.parent_scope));
        }
        visit::walk_attribute(self, attr);
    }

    fn visit_arm(&mut self, arm: &'a ast::Arm) {
        if arm.is_placeholder {
            self.visit_invoc(arm.id);
        } else {
            visit::walk_arm(self, arm);
        }
    }

    fn visit_expr_field(&mut self, f: &'a ast::ExprField) {
        if f.is_placeholder {
            self.visit_invoc(f.id);
        } else {
            visit::walk_expr_field(self, f);
        }
    }

    fn visit_pat_field(&mut self, fp: &'a ast::PatField) {
        if fp.is_placeholder {
            self.visit_invoc(fp.id);
        } else {
            visit::walk_pat_field(self, fp);
        }
    }

    fn visit_generic_param(&mut self, param: &'a ast::GenericParam) {
        if param.is_placeholder {
            self.visit_invoc(param.id);
        } else {
            visit::walk_generic_param(self, param);
        }
    }

    fn visit_param(&mut self, p: &'a ast::Param) {
        if p.is_placeholder {
            self.visit_invoc(p.id);
        } else {
            visit::walk_param(self, p);
        }
    }

    fn visit_field_def(&mut self, sf: &'a ast::FieldDef) {
        if sf.is_placeholder {
            self.visit_invoc(sf.id);
        } else {
            let vis = self.resolve_visibility(&sf.vis);
            self.r.feed_visibility(self.r.feed(sf.id), vis);
            visit::walk_field_def(self, sf);
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn visit_variant(&mut self, variant: &'a ast::Variant) {
        if variant.is_placeholder {
            self.visit_invoc_in_module(variant.id);
            return;
        }

        let parent = self.parent_scope.module;
        let expn_id = self.parent_scope.expansion;
        let ident = variant.ident;

        // Define a name in the type namespace.
        let feed = self.r.feed(variant.id);
        let def_id = feed.key();
        let vis = self.resolve_visibility(&variant.vis);
        self.r.define(parent, ident, TypeNS, (self.res(def_id), vis, variant.span, expn_id));
        self.r.feed_visibility(feed, vis);

        // If the variant is marked as non_exhaustive then lower the visibility to within the crate.
        let ctor_vis =
            if vis.is_public() && ast::attr::contains_name(&variant.attrs, sym::non_exhaustive) {
                ty::Visibility::Restricted(CRATE_DEF_ID)
            } else {
                vis
            };

        // Define a constructor name in the value namespace.
        if let Some(ctor_node_id) = variant.data.ctor_node_id() {
            let feed = self.r.feed(ctor_node_id);
            let ctor_def_id = feed.key();
            let ctor_res = self.res(ctor_def_id);
            self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, variant.span, expn_id));
            self.r.feed_visibility(feed, ctor_vis);
        }

        // Record field names for error reporting.
        self.insert_field_idents(def_id, variant.data.fields());
        self.insert_field_visibilities_local(def_id.to_def_id(), variant.data.fields());

        visit::walk_variant(self, variant);
    }

    fn visit_where_predicate(&mut self, p: &'a ast::WherePredicate) {
        if p.is_placeholder {
            self.visit_invoc(p.id);
        } else {
            visit::walk_where_predicate(self, p);
        }
    }

    fn visit_crate(&mut self, krate: &'a ast::Crate) {
        if krate.is_placeholder {
            self.visit_invoc_in_module(krate.id);
        } else {
            // Visit attributes after items for backward compatibility.
            // This way they can use `macro_rules` defined later.
            visit::walk_list!(self, visit_item, &krate.items);
            visit::walk_list!(self, visit_attribute, &krate.attrs);
            self.contains_macro_use(&krate.attrs);
        }
    }
}
