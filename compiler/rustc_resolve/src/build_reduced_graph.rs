//! After we obtain a fresh AST fragment from a macro, code in this module helps to integrate
//! that fragment into the module structures that are already partially built.
//!
//! Items from the fragment are placed into modules,
//! unexpanded macros in the fragment are visited and registered.
//! Imports are also considered items and placed into modules here, but not resolved yet.

use crate::def_collector::collect_definitions;
use crate::imports::{Import, ImportKind};
use crate::macros::{MacroRulesBinding, MacroRulesScope, MacroRulesScopeRef};
use crate::Namespace::{self, MacroNS, TypeNS, ValueNS};
use crate::{Determinacy, ExternPreludeEntry, Finalize, Module, ModuleKind, ModuleOrUniformRoot};
use crate::{
    MacroData, NameBinding, NameBindingKind, ParentScope, PathResult, PerNS, ResolutionError,
};
use crate::{Resolver, ResolverArenas, Segment, ToNameBinding, VisResolutionError};

use rustc_ast::visit::{self, AssocCtxt, Visitor};
use rustc_ast::{self as ast, AssocItem, AssocItemKind, MetaItemKind, StmtKind};
use rustc_ast::{Block, Fn, ForeignItem, ForeignItemKind, Impl, Item, ItemKind, NodeId};
use rustc_attr as attr;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability};
use rustc_expand::expand::AstFragment;
use rustc_hir::def::{self, *};
use rustc_hir::def_id::{DefId, LocalDefId, CRATE_DEF_ID};
use rustc_metadata::creader::LoadedMacro;
use rustc_middle::bug;
use rustc_middle::metadata::ModChild;
use rustc_middle::ty::{self, DefIdTree};
use rustc_session::cstore::CrateStore;
use rustc_span::hygiene::{ExpnId, LocalExpnId, MacroKind};
use rustc_span::source_map::{respan, Spanned};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::Span;

use std::cell::Cell;
use std::ptr;

type Res = def::Res<NodeId>;

impl<'a, Id: Into<DefId>> ToNameBinding<'a>
    for (Module<'a>, ty::Visibility<Id>, Span, LocalExpnId)
{
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Module(self.0),
            ambiguity: None,
            vis: self.1.to_def_id(),
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'a, Id: Into<DefId>> ToNameBinding<'a> for (Res, ty::Visibility<Id>, Span, LocalExpnId) {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Res(self.0, false),
            ambiguity: None,
            vis: self.1.to_def_id(),
            span: self.2,
            expansion: self.3,
        })
    }
}

struct IsMacroExport;

impl<'a> ToNameBinding<'a> for (Res, ty::Visibility, Span, LocalExpnId, IsMacroExport) {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Res(self.0, true),
            ambiguity: None,
            vis: self.1.to_def_id(),
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'a> Resolver<'a> {
    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    pub(crate) fn define<T>(&mut self, parent: Module<'a>, ident: Ident, ns: Namespace, def: T)
    where
        T: ToNameBinding<'a>,
    {
        let binding = def.to_name_binding(self.arenas);
        let key = self.new_key(ident, ns);
        if let Err(old_binding) = self.try_define(parent, key, binding) {
            self.report_conflict(parent, ident, ns, old_binding, &binding);
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
    pub fn get_nearest_non_block_module(&mut self, mut def_id: DefId) -> Module<'a> {
        loop {
            match self.get_module(def_id) {
                Some(module) => return module,
                None => def_id = self.parent(def_id),
            }
        }
    }

    pub fn expect_module(&mut self, def_id: DefId) -> Module<'a> {
        self.get_module(def_id).expect("argument `DefId` is not a module")
    }

    /// If `def_id` refers to a module (in resolver's sense, i.e. a module item, crate root, enum,
    /// or trait), then this function returns that module's resolver representation, otherwise it
    /// returns `None`.
    pub(crate) fn get_module(&mut self, def_id: DefId) -> Option<Module<'a>> {
        if let module @ Some(..) = self.module_map.get(&def_id) {
            return module.copied();
        }

        if !def_id.is_local() {
            let def_kind = self.cstore().def_kind(def_id);
            match def_kind {
                DefKind::Mod | DefKind::Enum | DefKind::Trait => {
                    let def_key = self.cstore().def_key(def_id);
                    let parent = def_key.parent.map(|index| {
                        self.get_nearest_non_block_module(DefId { index, krate: def_id.krate })
                    });
                    let name = if let Some(cnum) = def_id.as_crate_root() {
                        self.cstore().crate_name(cnum)
                    } else {
                        def_key.disambiguated_data.data.get_opt_name().expect("module without name")
                    };

                    Some(self.new_module(
                        parent,
                        ModuleKind::Def(def_kind, def_id, name),
                        self.cstore().module_expansion_untracked(def_id, &self.session),
                        self.cstore().get_span_untracked(def_id, &self.session),
                        // FIXME: Account for `#[no_implicit_prelude]` attributes.
                        parent.map_or(false, |module| module.no_implicit_prelude),
                    ))
                }
                _ => None,
            }
        } else {
            None
        }
    }

    pub(crate) fn expn_def_scope(&mut self, expn_id: ExpnId) -> Module<'a> {
        match expn_id.expn_data().macro_def_id {
            Some(def_id) => self.macro_def_scope(def_id),
            None => expn_id
                .as_local()
                .and_then(|expn_id| self.ast_transform_scopes.get(&expn_id))
                .unwrap_or(&self.graph_root),
        }
    }

    pub(crate) fn macro_def_scope(&mut self, def_id: DefId) -> Module<'a> {
        if let Some(id) = def_id.as_local() {
            self.local_macro_def_scopes[&id]
        } else {
            self.get_nearest_non_block_module(def_id)
        }
    }

    pub(crate) fn get_macro(&mut self, res: Res) -> Option<MacroData> {
        match res {
            Res::Def(DefKind::Macro(..), def_id) => Some(self.get_macro_by_def_id(def_id)),
            Res::NonMacroAttr(_) => {
                Some(MacroData { ext: self.non_macro_attr.clone(), macro_rules: false })
            }
            _ => None,
        }
    }

    pub(crate) fn get_macro_by_def_id(&mut self, def_id: DefId) -> MacroData {
        if let Some(macro_data) = self.macro_map.get(&def_id) {
            return macro_data.clone();
        }

        let (ext, macro_rules) = match self.cstore().load_macro_untracked(def_id, &self.session) {
            LoadedMacro::MacroDef(item, edition) => (
                Lrc::new(self.compile_macro(&item, edition).0),
                matches!(item.kind, ItemKind::MacroDef(def) if def.macro_rules),
            ),
            LoadedMacro::ProcMacro(extz) => (Lrc::new(extz), false),
        };

        let macro_data = MacroData { ext, macro_rules };
        self.macro_map.insert(def_id, macro_data.clone());
        macro_data
    }

    pub(crate) fn build_reduced_graph(
        &mut self,
        fragment: &AstFragment,
        parent_scope: ParentScope<'a>,
    ) -> MacroRulesScopeRef<'a> {
        collect_definitions(self, fragment, parent_scope.expansion);
        let mut visitor = BuildReducedGraphVisitor { r: self, parent_scope };
        fragment.visit_with(&mut visitor);
        visitor.parent_scope.macro_rules
    }

    pub(crate) fn build_reduced_graph_external(&mut self, module: Module<'a>) {
        for child in self.cstore().module_children_untracked(module.def_id(), self.session) {
            let parent_scope = ParentScope::module(module, self);
            BuildReducedGraphVisitor { r: self, parent_scope }
                .build_reduced_graph_for_external_crate_res(child);
        }
    }
}

struct BuildReducedGraphVisitor<'a, 'b> {
    r: &'b mut Resolver<'a>,
    parent_scope: ParentScope<'a>,
}

impl<'a> AsMut<Resolver<'a>> for BuildReducedGraphVisitor<'a, '_> {
    fn as_mut(&mut self) -> &mut Resolver<'a> {
        self.r
    }
}

impl<'a, 'b> BuildReducedGraphVisitor<'a, 'b> {
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
                        self.r.visibilities[&def_id.expect_local()]
                    }
                    // Otherwise, the visibility is restricted to the nearest parent `mod` item.
                    _ => ty::Visibility::Restricted(
                        self.parent_scope.module.nearest_parent_mod().expect_local(),
                    ),
                })
            }
            ast::VisibilityKind::Restricted { ref path, id, .. } => {
                // Make `PRIVATE_IN_PUBLIC` lint a hard error.
                self.r.has_pub_restricted = true;
                // For visibilities we are not ready to provide correct implementation of "uniform
                // paths" right now, so on 2018 edition we only allow module-relative paths for now.
                // On 2015 edition visibilities are resolved as crate-relative by default,
                // so we are prepending a root segment if necessary.
                let ident = path.segments.get(0).expect("empty path in visibility").ident;
                let crate_root = if ident.is_path_segment_keyword() {
                    None
                } else if ident.span.rust_2015() {
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
                    Some(TypeNS),
                    parent_scope,
                    finalize.then(|| Finalize::new(id, path.span)),
                    None,
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                        let res = module.res().expect("visibility resolved to unnamed block");
                        if finalize {
                            self.r.record_partial_res(id, PartialRes::new(res));
                        }
                        if module.is_normal() {
                            if res == Res::Err {
                                Ok(ty::Visibility::Public)
                            } else {
                                let vis = ty::Visibility::Restricted(res.def_id());
                                if self.r.is_accessible_from(vis, parent_scope.module) {
                                    Ok(vis.expect_local())
                                } else {
                                    Err(VisResolutionError::AncestorOnly(path.span))
                                }
                            }
                        } else {
                            expected_found_error(res)
                        }
                    }
                    PathResult::Module(..) => Err(VisResolutionError::ModuleOnly(path.span)),
                    PathResult::NonModule(partial_res) => {
                        expected_found_error(partial_res.base_res())
                    }
                    PathResult::Failed { span, label, suggestion, .. } => {
                        Err(VisResolutionError::FailedToResolve(span, label, suggestion))
                    }
                    PathResult::Indeterminate => Err(VisResolutionError::Indeterminate(path.span)),
                }
            }
        }
    }

    fn insert_field_names_local(&mut self, def_id: DefId, vdata: &ast::VariantData) {
        let field_names = vdata
            .fields()
            .iter()
            .map(|field| respan(field.span, field.ident.map_or(kw::Empty, |ident| ident.name)))
            .collect();
        self.insert_field_names(def_id, field_names);
    }

    fn insert_field_names(&mut self, def_id: DefId, field_names: Vec<Spanned<Symbol>>) {
        self.r.field_names.insert(def_id, field_names);
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
        kind: ImportKind<'a>,
        span: Span,
        id: NodeId,
        item: &ast::Item,
        root_span: Span,
        root_id: NodeId,
        vis: ty::Visibility,
    ) {
        let current_module = self.parent_scope.module;
        let import = self.r.arenas.alloc_import(Import {
            kind,
            parent_scope: self.parent_scope,
            module_path,
            imported_module: Cell::new(None),
            span,
            id,
            use_span: item.span,
            use_span_with_attributes: item.span_with_attributes(),
            has_attributes: !item.attrs.is_empty(),
            root_span,
            root_id,
            vis: Cell::new(Some(vis)),
            used: Cell::new(false),
        });

        self.r.indeterminate_imports.push(import);
        match import.kind {
            // Don't add unresolved underscore imports to modules
            ImportKind::Single { target: Ident { name: kw::Underscore, .. }, .. } => {}
            ImportKind::Single { target, type_ns_only, .. } => {
                self.r.per_ns(|this, ns| {
                    if !type_ns_only || ns == TypeNS {
                        let key = this.new_key(target, ns);
                        let mut resolution = this.resolution(current_module, key).borrow_mut();
                        resolution.add_single_import(import);
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
        // The whole `use` item
        item: &Item,
        vis: ty::Visibility,
        root_span: Span,
    ) {
        debug!(
            "build_reduced_graph_for_use_tree(parent_prefix={:?}, use_tree={:?}, nested={})",
            parent_prefix, use_tree, nested
        );

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
            Some(seg) if !seg.ident.is_path_segment_keyword() && seg.ident.span.rust_2015() => {
                Some(seg.ident.span.ctxt())
            }
            None if is_glob && use_tree.span.rust_2015() => Some(use_tree.span.ctxt()),
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
            ast::UseTreeKind::Simple(rename, id1, id2) => {
                let mut ident = use_tree.ident();
                let mut module_path = prefix;
                let mut source = module_path.pop().unwrap();
                let mut type_ns_only = false;

                self.r.visibilities.insert(self.r.local_def_id(id), vis);
                if id1 != ast::DUMMY_NODE_ID {
                    self.r.visibilities.insert(self.r.local_def_id(id1), vis);
                }
                if id2 != ast::DUMMY_NODE_ID {
                    self.r.visibilities.insert(self.r.local_def_id(id2), vis);
                }

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
                        source = module_path.pop().unwrap();
                        if rename.is_none() {
                            ident = source.ident;
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
                        // in `source` breaks `src/test/ui/imports/import-crate-var.rs`,
                        // while the current crate doesn't have a valid `crate_name`.
                        if crate_name != kw::Empty {
                            // `crate_name` should not be interpreted as relative.
                            module_path.push(Segment::from_ident_and_id(
                                Ident { name: kw::PathRoot, span: source.ident.span },
                                self.r.next_node_id(),
                            ));
                            source.ident.name = crate_name;
                        }
                        if rename.is_none() {
                            ident.name = crate_name;
                        }

                        self.r
                            .session
                            .struct_span_err(item.span, "`$crate` may not be imported")
                            .emit();
                    }
                }

                if ident.name == kw::Crate {
                    self.r.session.span_err(
                        ident.span,
                        "crate root imports need to be explicitly named: \
                         `use crate as name;`",
                    );
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
                    additional_ids: (id1, id2),
                };

                self.add_import(
                    module_path,
                    kind,
                    use_tree.span,
                    id,
                    item,
                    root_span,
                    item.id,
                    vis,
                );
            }
            ast::UseTreeKind::Glob => {
                let kind = ImportKind::Glob {
                    is_prelude: self.r.session.contains_name(&item.attrs, sym::prelude_import),
                    max_vis: Cell::new(None),
                };
                self.r.visibilities.insert(self.r.local_def_id(id), vis);
                self.add_import(prefix, kind, use_tree.span, id, item, root_span, item.id, vis);
            }
            ast::UseTreeKind::Nested(ref items) => {
                // Ensure there is at most one `self` in the list
                let self_spans = items
                    .iter()
                    .filter_map(|&(ref use_tree, _)| {
                        if let ast::UseTreeKind::Simple(..) = use_tree.kind {
                            if use_tree.ident().name == kw::SelfLower {
                                return Some(use_tree.span);
                            }
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
                        tree, id, &prefix, true, // The whole `use` item
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
                        kind: ast::UseTreeKind::Simple(
                            Some(Ident::new(kw::Underscore, new_span)),
                            ast::DUMMY_NODE_ID,
                            ast::DUMMY_NODE_ID,
                        ),
                        span: use_tree.span,
                    };
                    self.build_reduced_graph_for_use_tree(
                        // This particular use tree
                        &tree,
                        id,
                        &prefix,
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

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self, item: &'b Item) {
        let parent_scope = &self.parent_scope;
        let parent = parent_scope.module;
        let expansion = parent_scope.expansion;
        let ident = item.ident;
        let sp = item.span;
        let vis = self.resolve_visibility(&item.vis);
        let local_def_id = self.r.local_def_id(item.id);
        let def_id = local_def_id.to_def_id();

        self.r.visibilities.insert(local_def_id, vis);

        match item.kind {
            ItemKind::Use(ref use_tree) => {
                self.build_reduced_graph_for_use_tree(
                    // This particular use tree
                    use_tree,
                    item.id,
                    &[],
                    false,
                    // The whole `use` item
                    item,
                    vis,
                    use_tree.span,
                );
            }

            ItemKind::ExternCrate(orig_name) => {
                self.build_reduced_graph_for_extern_crate(
                    orig_name,
                    item,
                    local_def_id,
                    vis,
                    parent,
                );
            }

            ItemKind::Mod(..) => {
                let module = self.r.new_module(
                    Some(parent),
                    ModuleKind::Def(DefKind::Mod, def_id, ident.name),
                    expansion.to_expn_id(),
                    item.span,
                    parent.no_implicit_prelude
                        || self.r.session.contains_name(&item.attrs, sym::no_implicit_prelude),
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));

                // Descend into the module.
                self.parent_scope.module = module;
            }

            // These items live in the value namespace.
            ItemKind::Static(_, mt, _) => {
                let res = Res::Def(DefKind::Static(mt), def_id);
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));
            }
            ItemKind::Const(..) => {
                let res = Res::Def(DefKind::Const, def_id);
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));
            }
            ItemKind::Fn(..) => {
                let res = Res::Def(DefKind::Fn, def_id);
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));

                // Functions introducing procedural macros reserve a slot
                // in the macro namespace as well (see #52225).
                self.define_macro(item);
            }

            // These items live in the type namespace.
            ItemKind::TyAlias(..) => {
                let res = Res::Def(DefKind::TyAlias, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));
            }

            ItemKind::Enum(_, _) => {
                let module = self.r.new_module(
                    Some(parent),
                    ModuleKind::Def(DefKind::Enum, def_id, ident.name),
                    expansion.to_expn_id(),
                    item.span,
                    parent.no_implicit_prelude,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.parent_scope.module = module;
            }

            ItemKind::TraitAlias(..) => {
                let res = Res::Def(DefKind::TraitAlias, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));
            }

            // These items live in both the type and value namespaces.
            ItemKind::Struct(ref vdata, _) => {
                // Define a name in the type namespace.
                let res = Res::Def(DefKind::Struct, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));

                // Record field names for error reporting.
                self.insert_field_names_local(def_id, vdata);

                // If this is a tuple or unit struct, define a name
                // in the value namespace as well.
                if let Some(ctor_node_id) = vdata.ctor_id() {
                    // If the structure is marked as non_exhaustive then lower the visibility
                    // to within the crate.
                    let mut ctor_vis = if vis.is_public()
                        && self.r.session.contains_name(&item.attrs, sym::non_exhaustive)
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
                        if ctor_vis.is_at_least(field_vis, &*self.r) {
                            ctor_vis = field_vis;
                        }
                        ret_fields.push(field_vis.to_def_id());
                    }
                    let ctor_def_id = self.r.local_def_id(ctor_node_id);
                    let ctor_res = Res::Def(
                        DefKind::Ctor(CtorOf::Struct, CtorKind::from_ast(vdata)),
                        ctor_def_id.to_def_id(),
                    );
                    self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, sp, expansion));
                    self.r.visibilities.insert(ctor_def_id, ctor_vis);

                    self.r
                        .struct_constructors
                        .insert(def_id, (ctor_res, ctor_vis.to_def_id(), ret_fields));
                }
            }

            ItemKind::Union(ref vdata, _) => {
                let res = Res::Def(DefKind::Union, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));

                // Record field names for error reporting.
                self.insert_field_names_local(def_id, vdata);
            }

            ItemKind::Trait(..) => {
                // Add all the items within to a new module.
                let module = self.r.new_module(
                    Some(parent),
                    ModuleKind::Def(DefKind::Trait, def_id, ident.name),
                    expansion.to_expn_id(),
                    item.span,
                    parent.no_implicit_prelude,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.parent_scope.module = module;
            }

            // These items do not add names to modules.
            ItemKind::Impl(box Impl { of_trait: Some(..), .. }) => {
                self.r.trait_impl_items.insert(local_def_id);
            }
            ItemKind::Impl { .. } | ItemKind::ForeignMod(..) | ItemKind::GlobalAsm(..) => {}

            ItemKind::MacroDef(..) | ItemKind::MacCall(_) => unreachable!(),
        }
    }

    fn build_reduced_graph_for_extern_crate(
        &mut self,
        orig_name: Option<Symbol>,
        item: &Item,
        local_def_id: LocalDefId,
        vis: ty::Visibility,
        parent: Module<'a>,
    ) {
        let ident = item.ident;
        let sp = item.span;
        let parent_scope = self.parent_scope;
        let expansion = parent_scope.expansion;

        let (used, module, binding) = if orig_name.is_none() && ident.name == kw::SelfLower {
            self.r
                .session
                .struct_span_err(item.span, "`extern crate self;` requires renaming")
                .span_suggestion(
                    item.span,
                    "rename the `self` crate to be able to import it",
                    "extern crate self as name;",
                    Applicability::HasPlaceholders,
                )
                .emit();
            return;
        } else if orig_name == Some(kw::SelfLower) {
            Some(self.r.graph_root)
        } else {
            self.r.crate_loader.process_extern_crate(item, &self.r.definitions, local_def_id).map(
                |crate_id| {
                    self.r.extern_crate_map.insert(local_def_id, crate_id);
                    self.r.expect_module(crate_id.as_def_id())
                },
            )
        }
        .map(|module| {
            let used = self.process_macro_use_imports(item, module);
            let vis = ty::Visibility::<LocalDefId>::Public;
            let binding = (module, vis, sp, expansion).to_name_binding(self.r.arenas);
            (used, Some(ModuleOrUniformRoot::Module(module)), binding)
        })
        .unwrap_or((true, None, self.r.dummy_binding));
        let import = self.r.arenas.alloc_import(Import {
            kind: ImportKind::ExternCrate { source: orig_name, target: ident },
            root_id: item.id,
            id: item.id,
            parent_scope: self.parent_scope,
            imported_module: Cell::new(module),
            has_attributes: !item.attrs.is_empty(),
            use_span_with_attributes: item.span_with_attributes(),
            use_span: item.span,
            root_span: item.span,
            span: item.span,
            module_path: Vec::new(),
            vis: Cell::new(Some(vis)),
            used: Cell::new(used),
        });
        self.r.potentially_unused_imports.push(import);
        let imported_binding = self.r.import(binding, import);
        if ptr::eq(parent, self.r.graph_root) {
            if let Some(entry) = self.r.extern_prelude.get(&ident.normalize_to_macros_2_0()) {
                if expansion != LocalExpnId::ROOT
                    && orig_name.is_some()
                    && entry.extern_crate_item.is_none()
                {
                    let msg = "macro-expanded `extern crate` items cannot \
                                       shadow names passed with `--extern`";
                    self.r.session.span_err(item.span, msg);
                }
            }
            let entry = self.r.extern_prelude.entry(ident.normalize_to_macros_2_0()).or_insert(
                ExternPreludeEntry { extern_crate_item: None, introduced_by_item: true },
            );
            entry.extern_crate_item = Some(imported_binding);
            if orig_name.is_some() {
                entry.introduced_by_item = true;
            }
        }
        self.r.define(parent, ident, TypeNS, imported_binding);
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self, item: &ForeignItem) {
        let local_def_id = self.r.local_def_id(item.id);
        let def_id = local_def_id.to_def_id();
        let (def_kind, ns) = match item.kind {
            ForeignItemKind::Fn(..) => (DefKind::Fn, ValueNS),
            ForeignItemKind::Static(_, mt, _) => (DefKind::Static(mt), ValueNS),
            ForeignItemKind::TyAlias(..) => (DefKind::ForeignTy, TypeNS),
            ForeignItemKind::MacCall(_) => unreachable!(),
        };
        let parent = self.parent_scope.module;
        let expansion = self.parent_scope.expansion;
        let vis = self.resolve_visibility(&item.vis);
        let res = Res::Def(def_kind, def_id);
        self.r.define(parent, item.ident, ns, (res, vis, item.span, expansion));
        self.r.visibilities.insert(local_def_id, vis);
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

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_res(&mut self, child: ModChild) {
        let parent = self.parent_scope.module;
        let ModChild { ident, res, vis, span, macro_rules } = child;
        let res = res.expect_non_local();
        let expansion = self.parent_scope.expansion;
        // Record primary definitions.
        match res {
            Res::Def(DefKind::Mod | DefKind::Enum | DefKind::Trait, def_id) => {
                let module = self.r.expect_module(def_id);
                self.r.define(parent, ident, TypeNS, (module, vis, span, expansion));
            }
            Res::Def(
                DefKind::Struct
                | DefKind::Union
                | DefKind::Variant
                | DefKind::TyAlias
                | DefKind::ForeignTy
                | DefKind::OpaqueTy
                | DefKind::ImplTraitPlaceholder
                | DefKind::TraitAlias
                | DefKind::AssocTy,
                _,
            )
            | Res::PrimTy(..)
            | Res::ToolMod => self.r.define(parent, ident, TypeNS, (res, vis, span, expansion)),
            Res::Def(
                DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Static(_)
                | DefKind::Const
                | DefKind::AssocConst
                | DefKind::Ctor(..),
                _,
            ) => self.r.define(parent, ident, ValueNS, (res, vis, span, expansion)),
            Res::Def(DefKind::Macro(..), _) | Res::NonMacroAttr(..) => {
                if !macro_rules {
                    self.r.define(parent, ident, MacroNS, (res, vis, span, expansion))
                }
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
                | DefKind::Impl
                | DefKind::Generator,
                _,
            )
            | Res::Local(..)
            | Res::SelfTy { .. }
            | Res::SelfCtor(..)
            | Res::Err => bug!("unexpected resolution: {:?}", res),
        }
        // Record some extra data for better diagnostics.
        let cstore = self.r.cstore();
        match res {
            Res::Def(DefKind::Struct, def_id) => {
                let field_names =
                    cstore.struct_field_names_untracked(def_id, self.r.session).collect();
                let ctor = cstore.ctor_def_id_and_kind_untracked(def_id);
                if let Some((ctor_def_id, ctor_kind)) = ctor {
                    let ctor_res = Res::Def(DefKind::Ctor(CtorOf::Struct, ctor_kind), ctor_def_id);
                    let ctor_vis = cstore.visibility_untracked(ctor_def_id);
                    let field_visibilities =
                        cstore.struct_field_visibilities_untracked(def_id).collect();
                    self.r
                        .struct_constructors
                        .insert(def_id, (ctor_res, ctor_vis, field_visibilities));
                }
                self.insert_field_names(def_id, field_names);
            }
            Res::Def(DefKind::Union, def_id) => {
                let field_names =
                    cstore.struct_field_names_untracked(def_id, self.r.session).collect();
                self.insert_field_names(def_id, field_names);
            }
            Res::Def(DefKind::AssocFn, def_id) => {
                if cstore.fn_has_self_parameter_untracked(def_id, self.r.session) {
                    self.r.has_self.insert(def_id);
                }
            }
            _ => {}
        }
    }

    fn add_macro_use_binding(
        &mut self,
        name: Symbol,
        binding: &'a NameBinding<'a>,
        span: Span,
        allow_shadowing: bool,
    ) {
        if self.r.macro_use_prelude.insert(name, binding).is_some() && !allow_shadowing {
            let msg = format!("`{}` is already in scope", name);
            let note =
                "macro-expanded `#[macro_use]`s may not shadow existing macros (see RFC 1560)";
            self.r.session.struct_span_err(span, &msg).note(note).emit();
        }
    }

    /// Returns `true` if we should consider the underlying `extern crate` to be used.
    fn process_macro_use_imports(&mut self, item: &Item, module: Module<'a>) -> bool {
        let mut import_all = None;
        let mut single_imports = Vec::new();
        for attr in &item.attrs {
            if attr.has_name(sym::macro_use) {
                if self.parent_scope.module.parent.is_some() {
                    struct_span_err!(
                        self.r.session,
                        item.span,
                        E0468,
                        "an `extern crate` loading macros must be at the crate root"
                    )
                    .emit();
                }
                if let ItemKind::ExternCrate(Some(orig_name)) = item.kind {
                    if orig_name == kw::SelfLower {
                        self.r
                            .session
                            .struct_span_err(
                                attr.span,
                                "`#[macro_use]` is not supported on `extern crate self`",
                            )
                            .emit();
                    }
                }
                let ill_formed = |span| {
                    struct_span_err!(self.r.session, span, E0466, "bad macro import").emit();
                };
                match attr.meta() {
                    Some(meta) => match meta.kind {
                        MetaItemKind::Word => {
                            import_all = Some(meta.span);
                            break;
                        }
                        MetaItemKind::List(nested_metas) => {
                            for nested_meta in nested_metas {
                                match nested_meta.ident() {
                                    Some(ident) if nested_meta.is_word() => {
                                        single_imports.push(ident)
                                    }
                                    _ => ill_formed(nested_meta.span()),
                                }
                            }
                        }
                        MetaItemKind::NameValue(..) => ill_formed(meta.span),
                    },
                    None => ill_formed(attr.span),
                }
            }
        }

        let macro_use_import = |this: &Self, span| {
            this.r.arenas.alloc_import(Import {
                kind: ImportKind::MacroUse,
                root_id: item.id,
                id: item.id,
                parent_scope: this.parent_scope,
                imported_module: Cell::new(Some(ModuleOrUniformRoot::Module(module))),
                use_span_with_attributes: item.span_with_attributes(),
                has_attributes: !item.attrs.is_empty(),
                use_span: item.span,
                root_span: span,
                span,
                module_path: Vec::new(),
                vis: Cell::new(Some(ty::Visibility::Restricted(CRATE_DEF_ID))),
                used: Cell::new(false),
            })
        };

        let allow_shadowing = self.parent_scope.expansion == LocalExpnId::ROOT;
        if let Some(span) = import_all {
            let import = macro_use_import(self, span);
            self.r.potentially_unused_imports.push(import);
            module.for_each_child(self, |this, ident, ns, binding| {
                if ns == MacroNS {
                    let imported_binding = this.r.import(binding, import);
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
                );
                if let Ok(binding) = result {
                    let import = macro_use_import(self, ident.span);
                    self.r.potentially_unused_imports.push(import);
                    let imported_binding = self.r.import(binding, import);
                    self.add_macro_use_binding(
                        ident.name,
                        imported_binding,
                        ident.span,
                        allow_shadowing,
                    );
                } else {
                    struct_span_err!(self.r.session, ident.span, E0469, "imported macro not found")
                        .emit();
                }
            }
        }
        import_all.is_some() || !single_imports.is_empty()
    }

    /// Returns `true` if this attribute list contains `macro_use`.
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.has_name(sym::macro_escape) {
                let msg = "`#[macro_escape]` is a deprecated synonym for `#[macro_use]`";
                let mut err = self.r.session.struct_span_warn(attr.span, msg);
                if let ast::AttrStyle::Inner = attr.style {
                    err.help("try an outer attribute: `#[macro_use]`").emit();
                } else {
                    err.emit();
                }
            } else if !attr.has_name(sym::macro_use) {
                continue;
            }

            if !attr.is_word() {
                self.r.session.span_err(attr.span, "arguments to `macro_use` are not allowed here");
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
    fn visit_invoc_in_module(&mut self, id: NodeId) -> MacroRulesScopeRef<'a> {
        let invoc_id = self.visit_invoc(id);
        self.parent_scope.module.unexpanded_invocations.borrow_mut().insert(invoc_id);
        self.r.arenas.alloc_macro_rules_scope(MacroRulesScope::Invocation(invoc_id))
    }

    fn proc_macro_stub(&self, item: &ast::Item) -> Option<(MacroKind, Ident, Span)> {
        if self.r.session.contains_name(&item.attrs, sym::proc_macro) {
            return Some((MacroKind::Bang, item.ident, item.span));
        } else if self.r.session.contains_name(&item.attrs, sym::proc_macro_attribute) {
            return Some((MacroKind::Attr, item.ident, item.span));
        } else if let Some(attr) = self.r.session.find_by_name(&item.attrs, sym::proc_macro_derive)
        {
            if let Some(nested_meta) = attr.meta_item_list().and_then(|list| list.get(0).cloned()) {
                if let Some(ident) = nested_meta.ident() {
                    return Some((MacroKind::Derive, ident, ident.span));
                }
            }
        }
        None
    }

    // Mark the given macro as unused unless its name starts with `_`.
    // Macro uses will remove items from this set, and the remaining
    // items will be reported as `unused_macros`.
    fn insert_unused_macro(
        &mut self,
        ident: Ident,
        def_id: LocalDefId,
        node_id: NodeId,
        rule_spans: &[(usize, Span)],
    ) {
        if !ident.as_str().starts_with('_') {
            self.r.unused_macros.insert(def_id, (node_id, ident));
            for (rule_i, rule_span) in rule_spans.iter() {
                self.r.unused_macro_rules.insert((def_id, *rule_i), (ident, *rule_span));
            }
        }
    }

    fn define_macro(&mut self, item: &ast::Item) -> MacroRulesScopeRef<'a> {
        let parent_scope = self.parent_scope;
        let expansion = parent_scope.expansion;
        let def_id = self.r.local_def_id(item.id);
        let (ext, ident, span, macro_rules, rule_spans) = match &item.kind {
            ItemKind::MacroDef(def) => {
                let (ext, rule_spans) = self.r.compile_macro(item, self.r.session.edition());
                let ext = Lrc::new(ext);
                (ext, item.ident, item.span, def.macro_rules, rule_spans)
            }
            ItemKind::Fn(..) => match self.proc_macro_stub(item) {
                Some((macro_kind, ident, span)) => {
                    self.r.proc_macro_stubs.insert(def_id);
                    (self.r.dummy_ext(macro_kind), ident, span, false, Vec::new())
                }
                None => return parent_scope.macro_rules,
            },
            _ => unreachable!(),
        };

        let res = Res::Def(DefKind::Macro(ext.macro_kind()), def_id.to_def_id());
        self.r.macro_map.insert(def_id.to_def_id(), MacroData { ext, macro_rules });
        self.r.local_macro_def_scopes.insert(def_id, parent_scope.module);

        if macro_rules {
            let ident = ident.normalize_to_macros_2_0();
            self.r.macro_names.insert(ident);
            let is_macro_export = self.r.session.contains_name(&item.attrs, sym::macro_export);
            let vis = if is_macro_export {
                ty::Visibility::Public
            } else {
                ty::Visibility::Restricted(CRATE_DEF_ID)
            };
            let binding = (res, vis, span, expansion).to_name_binding(self.r.arenas);
            self.r.set_binding_parent_module(binding, parent_scope.module);
            if is_macro_export {
                let module = self.r.graph_root;
                self.r.define(module, ident, MacroNS, (res, vis, span, expansion, IsMacroExport));
            } else {
                self.r.check_reserved_macro_name(ident, res);
                self.insert_unused_macro(ident, def_id, item.id, &rule_spans);
            }
            self.r.visibilities.insert(def_id, vis);
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
                self.insert_unused_macro(ident, def_id, item.id, &rule_spans);
            }
            self.r.define(module, ident, MacroNS, (res, vis, span, expansion));
            self.r.visibilities.insert(def_id, vis);
            self.parent_scope.macro_rules
        }
    }
}

macro_rules! method {
    ($visit:ident: $ty:ty, $invoc:path, $walk:ident) => {
        fn $visit(&mut self, node: &'b $ty) {
            if let $invoc(..) = node.kind {
                self.visit_invoc(node.id);
            } else {
                visit::$walk(self, node);
            }
        }
    };
}

impl<'a, 'b> Visitor<'b> for BuildReducedGraphVisitor<'a, 'b> {
    method!(visit_expr: ast::Expr, ast::ExprKind::MacCall, walk_expr);
    method!(visit_pat: ast::Pat, ast::PatKind::MacCall, walk_pat);
    method!(visit_ty: ast::Ty, ast::TyKind::MacCall, walk_ty);

    fn visit_item(&mut self, item: &'b Item) {
        let orig_module_scope = self.parent_scope.module;
        self.parent_scope.macro_rules = match item.kind {
            ItemKind::MacroDef(..) => {
                let macro_rules_scope = self.define_macro(item);
                visit::walk_item(self, item);
                macro_rules_scope
            }
            ItemKind::MacCall(..) => {
                let macro_rules_scope = self.visit_invoc_in_module(item.id);
                visit::walk_item(self, item);
                macro_rules_scope
            }
            _ => {
                let orig_macro_rules_scope = self.parent_scope.macro_rules;
                self.build_reduced_graph_for_item(item);
                visit::walk_item(self, item);
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

    fn visit_stmt(&mut self, stmt: &'b ast::Stmt) {
        if let ast::StmtKind::MacCall(..) = stmt.kind {
            self.parent_scope.macro_rules = self.visit_invoc_in_module(stmt.id);
        } else {
            visit::walk_stmt(self, stmt);
        }
    }

    fn visit_foreign_item(&mut self, foreign_item: &'b ForeignItem) {
        if let ForeignItemKind::MacCall(_) = foreign_item.kind {
            self.visit_invoc_in_module(foreign_item.id);
            return;
        }

        self.build_reduced_graph_for_foreign_item(foreign_item);
        visit::walk_foreign_item(self, foreign_item);
    }

    fn visit_block(&mut self, block: &'b Block) {
        let orig_current_module = self.parent_scope.module;
        let orig_current_macro_rules_scope = self.parent_scope.macro_rules;
        self.build_reduced_graph_for_block(block);
        visit::walk_block(self, block);
        self.parent_scope.module = orig_current_module;
        self.parent_scope.macro_rules = orig_current_macro_rules_scope;
    }

    fn visit_assoc_item(&mut self, item: &'b AssocItem, ctxt: AssocCtxt) {
        if let AssocItemKind::MacCall(_) = item.kind {
            match ctxt {
                AssocCtxt::Trait => {
                    self.visit_invoc_in_module(item.id);
                }
                AssocCtxt::Impl => {
                    self.visit_invoc(item.id);
                }
            }
            return;
        }

        let vis = self.resolve_visibility(&item.vis);
        let local_def_id = self.r.local_def_id(item.id);
        let def_id = local_def_id.to_def_id();

        if !(ctxt == AssocCtxt::Impl
            && matches!(item.vis.kind, ast::VisibilityKind::Inherited)
            && self
                .r
                .trait_impl_items
                .contains(&ty::DefIdTree::local_parent(&*self.r, local_def_id)))
        {
            // Trait impl item visibility is inherited from its trait when not specified
            // explicitly. In that case we cannot determine it here in early resolve,
            // so we leave a hole in the visibility table to be filled later.
            self.r.visibilities.insert(local_def_id, vis);
        }

        if ctxt == AssocCtxt::Trait {
            let (def_kind, ns) = match item.kind {
                AssocItemKind::Const(..) => (DefKind::AssocConst, ValueNS),
                AssocItemKind::Fn(box Fn { ref sig, .. }) => {
                    if sig.decl.has_self() {
                        self.r.has_self.insert(def_id);
                    }
                    (DefKind::AssocFn, ValueNS)
                }
                AssocItemKind::TyAlias(..) => (DefKind::AssocTy, TypeNS),
                AssocItemKind::MacCall(_) => bug!(), // handled above
            };

            let parent = self.parent_scope.module;
            let expansion = self.parent_scope.expansion;
            let res = Res::Def(def_kind, def_id);
            self.r.define(parent, item.ident, ns, (res, vis, item.span, expansion));
        }

        visit::walk_assoc_item(self, item, ctxt);
    }

    fn visit_attribute(&mut self, attr: &'b ast::Attribute) {
        if !attr.is_doc_comment() && attr::is_builtin_attr(attr) {
            self.r
                .builtin_attrs
                .push((attr.get_normal_item().path.segments[0].ident, self.parent_scope));
        }
        visit::walk_attribute(self, attr);
    }

    fn visit_arm(&mut self, arm: &'b ast::Arm) {
        if arm.is_placeholder {
            self.visit_invoc(arm.id);
        } else {
            visit::walk_arm(self, arm);
        }
    }

    fn visit_expr_field(&mut self, f: &'b ast::ExprField) {
        if f.is_placeholder {
            self.visit_invoc(f.id);
        } else {
            visit::walk_expr_field(self, f);
        }
    }

    fn visit_pat_field(&mut self, fp: &'b ast::PatField) {
        if fp.is_placeholder {
            self.visit_invoc(fp.id);
        } else {
            visit::walk_pat_field(self, fp);
        }
    }

    fn visit_generic_param(&mut self, param: &'b ast::GenericParam) {
        if param.is_placeholder {
            self.visit_invoc(param.id);
        } else {
            visit::walk_generic_param(self, param);
        }
    }

    fn visit_param(&mut self, p: &'b ast::Param) {
        if p.is_placeholder {
            self.visit_invoc(p.id);
        } else {
            visit::walk_param(self, p);
        }
    }

    fn visit_field_def(&mut self, sf: &'b ast::FieldDef) {
        if sf.is_placeholder {
            self.visit_invoc(sf.id);
        } else {
            let vis = self.resolve_visibility(&sf.vis);
            self.r.visibilities.insert(self.r.local_def_id(sf.id), vis);
            visit::walk_field_def(self, sf);
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn visit_variant(&mut self, variant: &'b ast::Variant) {
        if variant.is_placeholder {
            self.visit_invoc_in_module(variant.id);
            return;
        }

        let parent = self.parent_scope.module;
        let expn_id = self.parent_scope.expansion;
        let ident = variant.ident;

        // Define a name in the type namespace.
        let def_id = self.r.local_def_id(variant.id);
        let res = Res::Def(DefKind::Variant, def_id.to_def_id());
        let vis = self.resolve_visibility(&variant.vis);
        self.r.define(parent, ident, TypeNS, (res, vis, variant.span, expn_id));
        self.r.visibilities.insert(def_id, vis);

        // If the variant is marked as non_exhaustive then lower the visibility to within the crate.
        let ctor_vis = if vis.is_public()
            && self.r.session.contains_name(&variant.attrs, sym::non_exhaustive)
        {
            ty::Visibility::Restricted(CRATE_DEF_ID)
        } else {
            vis
        };

        // Define a constructor name in the value namespace.
        // Braced variants, unlike structs, generate unusable names in
        // value namespace, they are reserved for possible future use.
        // It's ok to use the variant's id as a ctor id since an
        // error will be reported on any use of such resolution anyway.
        let ctor_node_id = variant.data.ctor_id().unwrap_or(variant.id);
        let ctor_def_id = self.r.local_def_id(ctor_node_id);
        let ctor_kind = CtorKind::from_ast(&variant.data);
        let ctor_res = Res::Def(DefKind::Ctor(CtorOf::Variant, ctor_kind), ctor_def_id.to_def_id());
        self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, variant.span, expn_id));
        if ctor_def_id != def_id {
            self.r.visibilities.insert(ctor_def_id, ctor_vis);
        }
        // Record field names for error reporting.
        self.insert_field_names_local(ctor_def_id.to_def_id(), &variant.data);

        visit::walk_variant(self, variant);
    }

    fn visit_crate(&mut self, krate: &'b ast::Crate) {
        if krate.is_placeholder {
            self.visit_invoc_in_module(krate.id);
        } else {
            visit::walk_crate(self, krate);
            self.contains_macro_use(&krate.attrs);
        }
    }
}
