//! After we obtain a fresh AST fragment from a macro, code in this module helps to integrate
//! that fragment into the module structures that are already partially built.
//!
//! Items from the fragment are placed into modules,
//! unexpanded macros in the fragment are visited and registered.
//! Imports are also considered items and placed into modules here, but not resolved yet.

use crate::def_collector::collect_definitions;
use crate::imports::ImportDirective;
use crate::imports::ImportDirectiveSubclass::{self, GlobImport, SingleImport};
use crate::macros::{LegacyBinding, LegacyScope};
use crate::Namespace::{self, MacroNS, TypeNS, ValueNS};
use crate::{CrateLint, Determinacy, PathResult, ResolutionError, VisResolutionError};
use crate::{
    ExternPreludeEntry, ModuleOrUniformRoot, ParentScope, PerNS, Resolver, ResolverArenas,
};
use crate::{Module, ModuleData, ModuleKind, NameBinding, NameBindingKind, Segment, ToNameBinding};

use rustc::bug;
use rustc::hir::def::{self, *};
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::middle::cstore::CrateStore;
use rustc::ty;
use rustc_metadata::creader::LoadedMacro;

use rustc_data_structures::sync::Lrc;
use std::cell::Cell;
use std::ptr;

use errors::Applicability;

use rustc_expand::base::SyntaxExtension;
use rustc_expand::expand::AstFragment;
use rustc_span::hygiene::{ExpnId, MacroKind};
use rustc_span::{Span, DUMMY_SP};
use syntax::ast::{self, Block, ForeignItem, ForeignItemKind, Item, ItemKind, NodeId};
use syntax::ast::{AssocItem, AssocItemKind, MetaItemKind, StmtKind};
use syntax::ast::{Ident, Name};
use syntax::attr;
use syntax::source_map::{respan, Spanned};
use syntax::span_err;
use syntax::symbol::{kw, sym};
use syntax::token::{self, Token};
use syntax::visit::{self, Visitor};

use log::debug;

use rustc_error_codes::*;

type Res = def::Res<NodeId>;

impl<'a> ToNameBinding<'a> for (Module<'a>, ty::Visibility, Span, ExpnId) {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Module(self.0),
            ambiguity: None,
            vis: self.1,
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'a> ToNameBinding<'a> for (Res, ty::Visibility, Span, ExpnId) {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Res(self.0, false),
            ambiguity: None,
            vis: self.1,
            span: self.2,
            expansion: self.3,
        })
    }
}

struct IsMacroExport;

impl<'a> ToNameBinding<'a> for (Res, ty::Visibility, Span, ExpnId, IsMacroExport) {
    fn to_name_binding(self, arenas: &'a ResolverArenas<'a>) -> &'a NameBinding<'a> {
        arenas.alloc_name_binding(NameBinding {
            kind: NameBindingKind::Res(self.0, true),
            ambiguity: None,
            vis: self.1,
            span: self.2,
            expansion: self.3,
        })
    }
}

impl<'a> Resolver<'a> {
    /// Defines `name` in namespace `ns` of module `parent` to be `def` if it is not yet defined;
    /// otherwise, reports an error.
    crate fn define<T>(&mut self, parent: Module<'a>, ident: Ident, ns: Namespace, def: T)
    where
        T: ToNameBinding<'a>,
    {
        let binding = def.to_name_binding(self.arenas);
        let key = self.new_key(ident, ns);
        if let Err(old_binding) = self.try_define(parent, key, binding) {
            self.report_conflict(parent, ident, ns, old_binding, &binding);
        }
    }

    crate fn get_module(&mut self, def_id: DefId) -> Module<'a> {
        if def_id.krate == LOCAL_CRATE {
            return self.module_map[&def_id];
        }

        if let Some(&module) = self.extern_module_map.get(&def_id) {
            return module;
        }

        let (name, parent) = if def_id.index == CRATE_DEF_INDEX {
            (self.cstore().crate_name_untracked(def_id.krate), None)
        } else {
            let def_key = self.cstore().def_key(def_id);
            (
                def_key.disambiguated_data.data.get_opt_name().unwrap(),
                Some(self.get_module(DefId { index: def_key.parent.unwrap(), ..def_id })),
            )
        };

        let kind = ModuleKind::Def(DefKind::Mod, def_id, name);
        let module = self.arenas.alloc_module(ModuleData::new(
            parent,
            kind,
            def_id,
            ExpnId::root(),
            DUMMY_SP,
        ));
        self.extern_module_map.insert(def_id, module);
        module
    }

    crate fn macro_def_scope(&mut self, expn_id: ExpnId) -> Module<'a> {
        let def_id = match self.macro_defs.get(&expn_id) {
            Some(def_id) => *def_id,
            None => return self.ast_transform_scopes.get(&expn_id).unwrap_or(&self.graph_root),
        };
        if let Some(id) = self.definitions.as_local_node_id(def_id) {
            self.local_macro_def_scopes[&id]
        } else {
            let module_def_id = ty::DefIdTree::parent(&*self, def_id).unwrap();
            self.get_module(module_def_id)
        }
    }

    crate fn get_macro(&mut self, res: Res) -> Option<Lrc<SyntaxExtension>> {
        match res {
            Res::Def(DefKind::Macro(..), def_id) => self.get_macro_by_def_id(def_id),
            Res::NonMacroAttr(attr_kind) => Some(self.non_macro_attr(attr_kind.is_used())),
            _ => None,
        }
    }

    crate fn get_macro_by_def_id(&mut self, def_id: DefId) -> Option<Lrc<SyntaxExtension>> {
        if let Some(ext) = self.macro_map.get(&def_id) {
            return Some(ext.clone());
        }

        let ext = Lrc::new(match self.cstore().load_macro_untracked(def_id, &self.session) {
            LoadedMacro::MacroDef(item, edition) => self.compile_macro(&item, edition),
            LoadedMacro::ProcMacro(ext) => ext,
        });

        self.macro_map.insert(def_id, ext.clone());
        Some(ext)
    }

    crate fn build_reduced_graph(
        &mut self,
        fragment: &AstFragment,
        parent_scope: ParentScope<'a>,
    ) -> LegacyScope<'a> {
        collect_definitions(&mut self.definitions, fragment, parent_scope.expansion);
        let mut visitor = BuildReducedGraphVisitor { r: self, parent_scope };
        fragment.visit_with(&mut visitor);
        visitor.parent_scope.legacy
    }

    crate fn build_reduced_graph_external(&mut self, module: Module<'a>) {
        let def_id = module.def_id().expect("unpopulated module without a def-id");
        for child in self.cstore().item_children_untracked(def_id, self.session) {
            let child = child.map_id(|_| panic!("unexpected id"));
            BuildReducedGraphVisitor { r: self, parent_scope: ParentScope::module(module) }
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
        self.resolve_visibility_speculative(vis, false).unwrap_or_else(|err| {
            self.r.report_vis_error(err);
            ty::Visibility::Public
        })
    }

    fn resolve_visibility_speculative<'ast>(
        &mut self,
        vis: &'ast ast::Visibility,
        speculative: bool,
    ) -> Result<ty::Visibility, VisResolutionError<'ast>> {
        let parent_scope = &self.parent_scope;
        match vis.node {
            ast::VisibilityKind::Public => Ok(ty::Visibility::Public),
            ast::VisibilityKind::Crate(..) => {
                Ok(ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX)))
            }
            ast::VisibilityKind::Inherited => {
                Ok(ty::Visibility::Restricted(parent_scope.module.normal_ancestor_id))
            }
            ast::VisibilityKind::Restricted { ref path, id, .. } => {
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
                    !speculative,
                    path.span,
                    CrateLint::SimplePath(id),
                ) {
                    PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                        let res = module.res().expect("visibility resolved to unnamed block");
                        if !speculative {
                            self.r.record_partial_res(id, PartialRes::new(res));
                        }
                        if module.is_normal() {
                            if res == Res::Err {
                                Ok(ty::Visibility::Public)
                            } else {
                                let vis = ty::Visibility::Restricted(res.def_id());
                                if self.r.is_accessible_from(vis, parent_scope.module) {
                                    Ok(vis)
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
            .map(|field| respan(field.span, field.ident.map_or(kw::Invalid, |ident| ident.name)))
            .collect();
        self.insert_field_names(def_id, field_names);
    }

    fn insert_field_names(&mut self, def_id: DefId, field_names: Vec<Spanned<Name>>) {
        if !field_names.is_empty() {
            self.r.field_names.insert(def_id, field_names);
        }
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        // If any statements are items, we need to create an anonymous module
        block.stmts.iter().any(|statement| match statement.kind {
            StmtKind::Item(_) | StmtKind::Mac(_) => true,
            _ => false,
        })
    }

    // Add an import directive to the current module.
    fn add_import_directive(
        &mut self,
        module_path: Vec<Segment>,
        subclass: ImportDirectiveSubclass<'a>,
        span: Span,
        id: NodeId,
        item: &ast::Item,
        root_span: Span,
        root_id: NodeId,
        vis: ty::Visibility,
    ) {
        let current_module = self.parent_scope.module;
        let directive = self.r.arenas.alloc_import_directive(ImportDirective {
            parent_scope: self.parent_scope,
            module_path,
            imported_module: Cell::new(None),
            subclass,
            span,
            id,
            use_span: item.span,
            use_span_with_attributes: item.span_with_attributes(),
            has_attributes: !item.attrs.is_empty(),
            root_span,
            root_id,
            vis: Cell::new(vis),
            used: Cell::new(false),
        });

        debug!("add_import_directive({:?})", directive);

        self.r.indeterminate_imports.push(directive);
        match directive.subclass {
            // Don't add unresolved underscore imports to modules
            SingleImport { target: Ident { name: kw::Underscore, .. }, .. } => {}
            SingleImport { target, type_ns_only, .. } => {
                self.r.per_ns(|this, ns| {
                    if !type_ns_only || ns == TypeNS {
                        let key = this.new_key(target, ns);
                        let mut resolution = this.resolution(current_module, key).borrow_mut();
                        resolution.add_single_import(directive);
                    }
                });
            }
            // We don't add prelude imports to the globs since they only affect lexical scopes,
            // which are not relevant to import resolution.
            GlobImport { is_prelude: true, .. } => {}
            GlobImport { .. } => current_module.globs.borrow_mut().push(directive),
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
        let is_glob = if let ast::UseTreeKind::Glob = use_tree.kind { true } else { false };
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
            ast::UseTreeKind::Simple(rename, ..) => {
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

                        // Replace `use foo::self;` with `use foo;`
                        source = module_path.pop().unwrap();
                        if rename.is_none() {
                            ident = source.ident;
                        }
                    }
                } else {
                    // Disallow `self`
                    if source.ident.name == kw::SelfLower {
                        self.r.report_error(
                            use_tree.span,
                            ResolutionError::SelfImportsOnlyAllowedWithin,
                        );
                    }

                    // Disallow `use $crate;`
                    if source.ident.name == kw::DollarCrate && module_path.is_empty() {
                        let crate_root = self.r.resolve_crate_root(source.ident);
                        let crate_name = match crate_root.kind {
                            ModuleKind::Def(.., name) => name,
                            ModuleKind::Block(..) => unreachable!(),
                        };
                        // HACK(eddyb) unclear how good this is, but keeping `$crate`
                        // in `source` breaks `src/test/compile-fail/import-crate-var.rs`,
                        // while the current crate doesn't have a valid `crate_name`.
                        if crate_name != kw::Invalid {
                            // `crate_name` should not be interpreted as relative.
                            module_path.push(Segment {
                                ident: Ident { name: kw::PathRoot, span: source.ident.span },
                                id: Some(self.r.next_node_id()),
                            });
                            source.ident.name = crate_name;
                        }
                        if rename.is_none() {
                            ident.name = crate_name;
                        }

                        self.r
                            .session
                            .struct_span_warn(item.span, "`$crate` may not be imported")
                            .note(
                                "`use $crate;` was erroneously allowed and \
                                   will become a hard error in a future release",
                            )
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

                let subclass = SingleImport {
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
                };
                self.add_import_directive(
                    module_path,
                    subclass,
                    use_tree.span,
                    id,
                    item,
                    root_span,
                    item.id,
                    vis,
                );
            }
            ast::UseTreeKind::Glob => {
                let subclass = GlobImport {
                    is_prelude: attr::contains_name(&item.attrs, sym::prelude_import),
                    max_vis: Cell::new(ty::Visibility::Invisible),
                };
                self.add_import_directive(
                    prefix,
                    subclass,
                    use_tree.span,
                    id,
                    item,
                    root_span,
                    item.id,
                    vis,
                );
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
                        ty::Visibility::Invisible,
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
                let module = if orig_name.is_none() && ident.name == kw::SelfLower {
                    self.r
                        .session
                        .struct_span_err(item.span, "`extern crate self;` requires renaming")
                        .span_suggestion(
                            item.span,
                            "try",
                            "extern crate self as name;".into(),
                            Applicability::HasPlaceholders,
                        )
                        .emit();
                    return;
                } else if orig_name == Some(kw::SelfLower) {
                    self.r.graph_root
                } else {
                    let crate_id =
                        self.r.crate_loader.process_extern_crate(item, &self.r.definitions);
                    self.r.extern_crate_map.insert(item.id, crate_id);
                    self.r.get_module(DefId { krate: crate_id, index: CRATE_DEF_INDEX })
                };

                let used = self.process_legacy_macro_imports(item, module);
                let binding =
                    (module, ty::Visibility::Public, sp, expansion).to_name_binding(self.r.arenas);
                let directive = self.r.arenas.alloc_import_directive(ImportDirective {
                    root_id: item.id,
                    id: item.id,
                    parent_scope: self.parent_scope,
                    imported_module: Cell::new(Some(ModuleOrUniformRoot::Module(module))),
                    subclass: ImportDirectiveSubclass::ExternCrate {
                        source: orig_name,
                        target: ident,
                    },
                    has_attributes: !item.attrs.is_empty(),
                    use_span_with_attributes: item.span_with_attributes(),
                    use_span: item.span,
                    root_span: item.span,
                    span: item.span,
                    module_path: Vec::new(),
                    vis: Cell::new(vis),
                    used: Cell::new(used),
                });
                self.r.potentially_unused_imports.push(directive);
                let imported_binding = self.r.import(binding, directive);
                if ptr::eq(parent, self.r.graph_root) {
                    if let Some(entry) = self.r.extern_prelude.get(&ident.modern()) {
                        if expansion != ExpnId::root()
                            && orig_name.is_some()
                            && entry.extern_crate_item.is_none()
                        {
                            let msg = "macro-expanded `extern crate` items cannot \
                                       shadow names passed with `--extern`";
                            self.r.session.span_err(item.span, msg);
                        }
                    }
                    let entry =
                        self.r.extern_prelude.entry(ident.modern()).or_insert(ExternPreludeEntry {
                            extern_crate_item: None,
                            introduced_by_item: true,
                        });
                    entry.extern_crate_item = Some(imported_binding);
                    if orig_name.is_some() {
                        entry.introduced_by_item = true;
                    }
                }
                self.r.define(parent, ident, TypeNS, imported_binding);
            }

            ItemKind::Mod(..) if ident.name == kw::Invalid => {} // Crate root

            ItemKind::Mod(..) => {
                let def_id = self.r.definitions.local_def_id(item.id);
                let module_kind = ModuleKind::Def(DefKind::Mod, def_id, ident.name);
                let module = self.r.arenas.alloc_module(ModuleData {
                    no_implicit_prelude: parent.no_implicit_prelude || {
                        attr::contains_name(&item.attrs, sym::no_implicit_prelude)
                    },
                    ..ModuleData::new(Some(parent), module_kind, def_id, expansion, item.span)
                });
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.r.module_map.insert(def_id, module);

                // Descend into the module.
                self.parent_scope.module = module;
            }

            // These items live in the value namespace.
            ItemKind::Static(..) => {
                let res = Res::Def(DefKind::Static, self.r.definitions.local_def_id(item.id));
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));
            }
            ItemKind::Const(..) => {
                let res = Res::Def(DefKind::Const, self.r.definitions.local_def_id(item.id));
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));
            }
            ItemKind::Fn(..) => {
                let res = Res::Def(DefKind::Fn, self.r.definitions.local_def_id(item.id));
                self.r.define(parent, ident, ValueNS, (res, vis, sp, expansion));

                // Functions introducing procedural macros reserve a slot
                // in the macro namespace as well (see #52225).
                self.define_macro(item);
            }

            // These items live in the type namespace.
            ItemKind::TyAlias(ref ty, _) => {
                let def_kind = match ty.kind.opaque_top_hack() {
                    None => DefKind::TyAlias,
                    Some(_) => DefKind::OpaqueTy,
                };
                let res = Res::Def(def_kind, self.r.definitions.local_def_id(item.id));
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));
            }

            ItemKind::Enum(_, _) => {
                let def_id = self.r.definitions.local_def_id(item.id);
                self.r.variant_vis.insert(def_id, vis);
                let module_kind = ModuleKind::Def(DefKind::Enum, def_id, ident.name);
                let module = self.r.new_module(
                    parent,
                    module_kind,
                    parent.normal_ancestor_id,
                    expansion,
                    item.span,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.parent_scope.module = module;
            }

            ItemKind::TraitAlias(..) => {
                let res = Res::Def(DefKind::TraitAlias, self.r.definitions.local_def_id(item.id));
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));
            }

            // These items live in both the type and value namespaces.
            ItemKind::Struct(ref vdata, _) => {
                // Define a name in the type namespace.
                let def_id = self.r.definitions.local_def_id(item.id);
                let res = Res::Def(DefKind::Struct, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));

                // Record field names for error reporting.
                self.insert_field_names_local(def_id, vdata);

                // If this is a tuple or unit struct, define a name
                // in the value namespace as well.
                if let Some(ctor_node_id) = vdata.ctor_id() {
                    let mut ctor_vis = vis;
                    // If the structure is marked as non_exhaustive then lower the visibility
                    // to within the crate.
                    if vis == ty::Visibility::Public
                        && attr::contains_name(&item.attrs, sym::non_exhaustive)
                    {
                        ctor_vis = ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
                    }
                    for field in vdata.fields() {
                        // NOTE: The field may be an expansion placeholder, but expansion sets
                        // correct visibilities for unnamed field placeholders specifically, so the
                        // constructor visibility should still be determined correctly.
                        if let Ok(field_vis) = self.resolve_visibility_speculative(&field.vis, true)
                        {
                            if ctor_vis.is_at_least(field_vis, &*self.r) {
                                ctor_vis = field_vis;
                            }
                        }
                    }
                    let ctor_res = Res::Def(
                        DefKind::Ctor(CtorOf::Struct, CtorKind::from_ast(vdata)),
                        self.r.definitions.local_def_id(ctor_node_id),
                    );
                    self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, sp, expansion));
                    self.r.struct_constructors.insert(def_id, (ctor_res, ctor_vis));
                }
            }

            ItemKind::Union(ref vdata, _) => {
                let def_id = self.r.definitions.local_def_id(item.id);
                let res = Res::Def(DefKind::Union, def_id);
                self.r.define(parent, ident, TypeNS, (res, vis, sp, expansion));

                // Record field names for error reporting.
                self.insert_field_names_local(def_id, vdata);
            }

            ItemKind::Trait(..) => {
                let def_id = self.r.definitions.local_def_id(item.id);

                // Add all the items within to a new module.
                let module_kind = ModuleKind::Def(DefKind::Trait, def_id, ident.name);
                let module = self.r.new_module(
                    parent,
                    module_kind,
                    parent.normal_ancestor_id,
                    expansion,
                    item.span,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, sp, expansion));
                self.parent_scope.module = module;
            }

            // These items do not add names to modules.
            ItemKind::Impl(..) | ItemKind::ForeignMod(..) | ItemKind::GlobalAsm(..) => {}

            ItemKind::MacroDef(..) | ItemKind::Mac(_) => unreachable!(),
        }
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self, item: &ForeignItem) {
        let (res, ns) = match item.kind {
            ForeignItemKind::Fn(..) => {
                (Res::Def(DefKind::Fn, self.r.definitions.local_def_id(item.id)), ValueNS)
            }
            ForeignItemKind::Static(..) => {
                (Res::Def(DefKind::Static, self.r.definitions.local_def_id(item.id)), ValueNS)
            }
            ForeignItemKind::Ty => {
                (Res::Def(DefKind::ForeignTy, self.r.definitions.local_def_id(item.id)), TypeNS)
            }
            ForeignItemKind::Macro(_) => unreachable!(),
        };
        let parent = self.parent_scope.module;
        let expansion = self.parent_scope.expansion;
        let vis = self.resolve_visibility(&item.vis);
        self.r.define(parent, item.ident, ns, (res, vis, item.span, expansion));
    }

    fn build_reduced_graph_for_block(&mut self, block: &Block) {
        let parent = self.parent_scope.module;
        let expansion = self.parent_scope.expansion;
        if self.block_needs_anonymous_module(block) {
            let module = self.r.new_module(
                parent,
                ModuleKind::Block(block.id),
                parent.normal_ancestor_id,
                expansion,
                block.span,
            );
            self.r.block_map.insert(block.id, module);
            self.parent_scope.module = module; // Descend into the block.
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_res(&mut self, child: Export<NodeId>) {
        let parent = self.parent_scope.module;
        let Export { ident, res, vis, span } = child;
        let expansion = ExpnId::root(); // FIXME(jseyfried) intercrate hygiene
        // Record primary definitions.
        match res {
            Res::Def(kind @ DefKind::Mod, def_id)
            | Res::Def(kind @ DefKind::Enum, def_id)
            | Res::Def(kind @ DefKind::Trait, def_id) => {
                let module = self.r.new_module(
                    parent,
                    ModuleKind::Def(kind, def_id, ident.name),
                    def_id,
                    expansion,
                    span,
                );
                self.r.define(parent, ident, TypeNS, (module, vis, span, expansion));
            }
            Res::Def(DefKind::Struct, _)
            | Res::Def(DefKind::Union, _)
            | Res::Def(DefKind::Variant, _)
            | Res::Def(DefKind::TyAlias, _)
            | Res::Def(DefKind::ForeignTy, _)
            | Res::Def(DefKind::OpaqueTy, _)
            | Res::Def(DefKind::TraitAlias, _)
            | Res::Def(DefKind::AssocTy, _)
            | Res::Def(DefKind::AssocOpaqueTy, _)
            | Res::PrimTy(..)
            | Res::ToolMod => self.r.define(parent, ident, TypeNS, (res, vis, span, expansion)),
            Res::Def(DefKind::Fn, _)
            | Res::Def(DefKind::Method, _)
            | Res::Def(DefKind::Static, _)
            | Res::Def(DefKind::Const, _)
            | Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::Ctor(..), _) => {
                self.r.define(parent, ident, ValueNS, (res, vis, span, expansion))
            }
            Res::Def(DefKind::Macro(..), _) | Res::NonMacroAttr(..) => {
                self.r.define(parent, ident, MacroNS, (res, vis, span, expansion))
            }
            Res::Def(DefKind::TyParam, _)
            | Res::Def(DefKind::ConstParam, _)
            | Res::Local(..)
            | Res::SelfTy(..)
            | Res::SelfCtor(..)
            | Res::Err => bug!("unexpected resolution: {:?}", res),
        }
        // Record some extra data for better diagnostics.
        let cstore = self.r.cstore();
        match res {
            Res::Def(DefKind::Struct, def_id) | Res::Def(DefKind::Union, def_id) => {
                let field_names = cstore.struct_field_names_untracked(def_id, self.r.session);
                self.insert_field_names(def_id, field_names);
            }
            Res::Def(DefKind::Method, def_id) => {
                if cstore.associated_item_cloned_untracked(def_id).method_has_self_argument {
                    self.r.has_self.insert(def_id);
                }
            }
            Res::Def(DefKind::Ctor(CtorOf::Struct, ..), def_id) => {
                let parent = cstore.def_key(def_id).parent;
                if let Some(struct_def_id) = parent.map(|index| DefId { index, ..def_id }) {
                    self.r.struct_constructors.insert(struct_def_id, (res, vis));
                }
            }
            _ => {}
        }
    }

    fn legacy_import_macro(
        &mut self,
        name: ast::Name,
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
    fn process_legacy_macro_imports(&mut self, item: &Item, module: Module<'a>) -> bool {
        let mut import_all = None;
        let mut single_imports = Vec::new();
        for attr in &item.attrs {
            if attr.check_name(sym::macro_use) {
                if self.parent_scope.module.parent.is_some() {
                    span_err!(
                        self.r.session,
                        item.span,
                        E0468,
                        "an `extern crate` loading macros must be at the crate root"
                    );
                }
                if let ItemKind::ExternCrate(Some(orig_name)) = item.kind {
                    if orig_name == kw::SelfLower {
                        self.r.session.span_err(
                            attr.span,
                            "`macro_use` is not supported on `extern crate self`",
                        );
                    }
                }
                let ill_formed = |span| span_err!(self.r.session, span, E0466, "bad macro import");
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

        let macro_use_directive = |this: &Self, span| {
            this.r.arenas.alloc_import_directive(ImportDirective {
                root_id: item.id,
                id: item.id,
                parent_scope: this.parent_scope,
                imported_module: Cell::new(Some(ModuleOrUniformRoot::Module(module))),
                subclass: ImportDirectiveSubclass::MacroUse,
                use_span_with_attributes: item.span_with_attributes(),
                has_attributes: !item.attrs.is_empty(),
                use_span: item.span,
                root_span: span,
                span,
                module_path: Vec::new(),
                vis: Cell::new(ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX))),
                used: Cell::new(false),
            })
        };

        let allow_shadowing = self.parent_scope.expansion == ExpnId::root();
        if let Some(span) = import_all {
            let directive = macro_use_directive(self, span);
            self.r.potentially_unused_imports.push(directive);
            module.for_each_child(self, |this, ident, ns, binding| {
                if ns == MacroNS {
                    let imported_binding = this.r.import(binding, directive);
                    this.legacy_import_macro(ident.name, imported_binding, span, allow_shadowing);
                }
            });
        } else {
            for ident in single_imports.iter().cloned() {
                let result = self.r.resolve_ident_in_module(
                    ModuleOrUniformRoot::Module(module),
                    ident,
                    MacroNS,
                    &self.parent_scope,
                    false,
                    ident.span,
                );
                if let Ok(binding) = result {
                    let directive = macro_use_directive(self, ident.span);
                    self.r.potentially_unused_imports.push(directive);
                    let imported_binding = self.r.import(binding, directive);
                    self.legacy_import_macro(
                        ident.name,
                        imported_binding,
                        ident.span,
                        allow_shadowing,
                    );
                } else {
                    span_err!(self.r.session, ident.span, E0469, "imported macro not found");
                }
            }
        }
        import_all.is_some() || !single_imports.is_empty()
    }

    /// Returns `true` if this attribute list contains `macro_use`.
    fn contains_macro_use(&mut self, attrs: &[ast::Attribute]) -> bool {
        for attr in attrs {
            if attr.check_name(sym::macro_escape) {
                let msg = "macro_escape is a deprecated synonym for macro_use";
                let mut err = self.r.session.struct_span_warn(attr.span, msg);
                if let ast::AttrStyle::Inner = attr.style {
                    err.help("consider an outer attribute, `#[macro_use]` mod ...").emit();
                } else {
                    err.emit();
                }
            } else if !attr.check_name(sym::macro_use) {
                continue;
            }

            if !attr.is_word() {
                self.r.session.span_err(attr.span, "arguments to macro_use are not allowed here");
            }
            return true;
        }

        false
    }

    fn visit_invoc(&mut self, id: NodeId) -> LegacyScope<'a> {
        let invoc_id = id.placeholder_to_expn_id();

        self.parent_scope.module.unexpanded_invocations.borrow_mut().insert(invoc_id);

        let old_parent_scope = self.r.invocation_parent_scopes.insert(invoc_id, self.parent_scope);
        assert!(old_parent_scope.is_none(), "invocation data is reset for an invocation");

        LegacyScope::Invocation(invoc_id)
    }

    fn proc_macro_stub(item: &ast::Item) -> Option<(MacroKind, Ident, Span)> {
        if attr::contains_name(&item.attrs, sym::proc_macro) {
            return Some((MacroKind::Bang, item.ident, item.span));
        } else if attr::contains_name(&item.attrs, sym::proc_macro_attribute) {
            return Some((MacroKind::Attr, item.ident, item.span));
        } else if let Some(attr) = attr::find_by_name(&item.attrs, sym::proc_macro_derive) {
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
    fn insert_unused_macro(&mut self, ident: Ident, node_id: NodeId, span: Span) {
        if !ident.as_str().starts_with("_") {
            self.r.unused_macros.insert(node_id, span);
        }
    }

    fn define_macro(&mut self, item: &ast::Item) -> LegacyScope<'a> {
        let parent_scope = self.parent_scope;
        let expansion = parent_scope.expansion;
        let (ext, ident, span, is_legacy) = match &item.kind {
            ItemKind::MacroDef(def) => {
                let ext = Lrc::new(self.r.compile_macro(item, self.r.session.edition()));
                (ext, item.ident, item.span, def.legacy)
            }
            ItemKind::Fn(..) => match Self::proc_macro_stub(item) {
                Some((macro_kind, ident, span)) => {
                    self.r.proc_macro_stubs.insert(item.id);
                    (self.r.dummy_ext(macro_kind), ident, span, false)
                }
                None => return parent_scope.legacy,
            },
            _ => unreachable!(),
        };

        let def_id = self.r.definitions.local_def_id(item.id);
        let res = Res::Def(DefKind::Macro(ext.macro_kind()), def_id);
        self.r.macro_map.insert(def_id, ext);
        self.r.local_macro_def_scopes.insert(item.id, parent_scope.module);

        if is_legacy {
            let ident = ident.modern();
            self.r.macro_names.insert(ident);
            let is_macro_export = attr::contains_name(&item.attrs, sym::macro_export);
            let vis = if is_macro_export {
                ty::Visibility::Public
            } else {
                ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX))
            };
            let binding = (res, vis, span, expansion).to_name_binding(self.r.arenas);
            self.r.set_binding_parent_module(binding, parent_scope.module);
            self.r.all_macros.insert(ident.name, res);
            if is_macro_export {
                let module = self.r.graph_root;
                self.r.define(module, ident, MacroNS, (res, vis, span, expansion, IsMacroExport));
            } else {
                self.r.check_reserved_macro_name(ident, res);
                self.insert_unused_macro(ident, item.id, span);
            }
            LegacyScope::Binding(self.r.arenas.alloc_legacy_binding(LegacyBinding {
                parent_legacy_scope: parent_scope.legacy,
                binding,
                ident,
            }))
        } else {
            let module = parent_scope.module;
            let vis = self.resolve_visibility(&item.vis);
            if vis != ty::Visibility::Public {
                self.insert_unused_macro(ident, item.id, span);
            }
            self.r.define(module, ident, MacroNS, (res, vis, span, expansion));
            self.parent_scope.legacy
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
    }
}

impl<'a, 'b> Visitor<'b> for BuildReducedGraphVisitor<'a, 'b> {
    method!(visit_expr: ast::Expr, ast::ExprKind::Mac, walk_expr);
    method!(visit_pat: ast::Pat, ast::PatKind::Mac, walk_pat);
    method!(visit_ty: ast::Ty, ast::TyKind::Mac, walk_ty);

    fn visit_item(&mut self, item: &'b Item) {
        let macro_use = match item.kind {
            ItemKind::MacroDef(..) => {
                self.parent_scope.legacy = self.define_macro(item);
                return;
            }
            ItemKind::Mac(..) => {
                self.parent_scope.legacy = self.visit_invoc(item.id);
                return;
            }
            ItemKind::Mod(..) => self.contains_macro_use(&item.attrs),
            _ => false,
        };
        let orig_current_module = self.parent_scope.module;
        let orig_current_legacy_scope = self.parent_scope.legacy;
        self.build_reduced_graph_for_item(item);
        visit::walk_item(self, item);
        self.parent_scope.module = orig_current_module;
        if !macro_use {
            self.parent_scope.legacy = orig_current_legacy_scope;
        }
    }

    fn visit_stmt(&mut self, stmt: &'b ast::Stmt) {
        if let ast::StmtKind::Mac(..) = stmt.kind {
            self.parent_scope.legacy = self.visit_invoc(stmt.id);
        } else {
            visit::walk_stmt(self, stmt);
        }
    }

    fn visit_foreign_item(&mut self, foreign_item: &'b ForeignItem) {
        if let ForeignItemKind::Macro(_) = foreign_item.kind {
            self.visit_invoc(foreign_item.id);
            return;
        }

        self.build_reduced_graph_for_foreign_item(foreign_item);
        visit::walk_foreign_item(self, foreign_item);
    }

    fn visit_block(&mut self, block: &'b Block) {
        let orig_current_module = self.parent_scope.module;
        let orig_current_legacy_scope = self.parent_scope.legacy;
        self.build_reduced_graph_for_block(block);
        visit::walk_block(self, block);
        self.parent_scope.module = orig_current_module;
        self.parent_scope.legacy = orig_current_legacy_scope;
    }

    fn visit_trait_item(&mut self, item: &'b AssocItem) {
        let parent = self.parent_scope.module;

        if let AssocItemKind::Macro(_) = item.kind {
            self.visit_invoc(item.id);
            return;
        }

        // Add the item to the trait info.
        let item_def_id = self.r.definitions.local_def_id(item.id);
        let (res, ns) = match item.kind {
            AssocItemKind::Const(..) => (Res::Def(DefKind::AssocConst, item_def_id), ValueNS),
            AssocItemKind::Fn(ref sig, _) => {
                if sig.decl.has_self() {
                    self.r.has_self.insert(item_def_id);
                }
                (Res::Def(DefKind::Method, item_def_id), ValueNS)
            }
            AssocItemKind::TyAlias(..) => (Res::Def(DefKind::AssocTy, item_def_id), TypeNS),
            AssocItemKind::Macro(_) => bug!(), // handled above
        };

        let vis = ty::Visibility::Public;
        let expansion = self.parent_scope.expansion;
        self.r.define(parent, item.ident, ns, (res, vis, item.span, expansion));

        visit::walk_trait_item(self, item);
    }

    fn visit_impl_item(&mut self, item: &'b ast::AssocItem) {
        if let ast::AssocItemKind::Macro(..) = item.kind {
            self.visit_invoc(item.id);
        } else {
            self.resolve_visibility(&item.vis);
            visit::walk_impl_item(self, item);
        }
    }

    fn visit_token(&mut self, t: Token) {
        if let token::Interpolated(nt) = t.kind {
            if let token::NtExpr(ref expr) = *nt {
                if let ast::ExprKind::Mac(..) = expr.kind {
                    self.visit_invoc(expr.id);
                }
            }
        }
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

    fn visit_field(&mut self, f: &'b ast::Field) {
        if f.is_placeholder {
            self.visit_invoc(f.id);
        } else {
            visit::walk_field(self, f);
        }
    }

    fn visit_field_pattern(&mut self, fp: &'b ast::FieldPat) {
        if fp.is_placeholder {
            self.visit_invoc(fp.id);
        } else {
            visit::walk_field_pattern(self, fp);
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

    fn visit_struct_field(&mut self, sf: &'b ast::StructField) {
        if sf.is_placeholder {
            self.visit_invoc(sf.id);
        } else {
            self.resolve_visibility(&sf.vis);
            visit::walk_struct_field(self, sf);
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and value namespaces.
    fn visit_variant(&mut self, variant: &'b ast::Variant) {
        if variant.is_placeholder {
            self.visit_invoc(variant.id);
            return;
        }

        let parent = self.parent_scope.module;
        let vis = self.r.variant_vis[&parent.def_id().expect("enum without def-id")];
        let expn_id = self.parent_scope.expansion;
        let ident = variant.ident;

        // Define a name in the type namespace.
        let def_id = self.r.definitions.local_def_id(variant.id);
        let res = Res::Def(DefKind::Variant, def_id);
        self.r.define(parent, ident, TypeNS, (res, vis, variant.span, expn_id));

        // If the variant is marked as non_exhaustive then lower the visibility to within the
        // crate.
        let mut ctor_vis = vis;
        let has_non_exhaustive = attr::contains_name(&variant.attrs, sym::non_exhaustive);
        if has_non_exhaustive && vis == ty::Visibility::Public {
            ctor_vis = ty::Visibility::Restricted(DefId::local(CRATE_DEF_INDEX));
        }

        // Define a constructor name in the value namespace.
        // Braced variants, unlike structs, generate unusable names in
        // value namespace, they are reserved for possible future use.
        // It's ok to use the variant's id as a ctor id since an
        // error will be reported on any use of such resolution anyway.
        let ctor_node_id = variant.data.ctor_id().unwrap_or(variant.id);
        let ctor_def_id = self.r.definitions.local_def_id(ctor_node_id);
        let ctor_kind = CtorKind::from_ast(&variant.data);
        let ctor_res = Res::Def(DefKind::Ctor(CtorOf::Variant, ctor_kind), ctor_def_id);
        self.r.define(parent, ident, ValueNS, (ctor_res, ctor_vis, variant.span, expn_id));

        visit::walk_variant(self, variant);
    }
}
