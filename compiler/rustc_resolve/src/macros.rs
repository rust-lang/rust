//! A bunch of methods and structures more or less related to resolving macros and
//! interface provided by `Resolver` to macro expander.

use std::cell::Cell;
use std::mem;
use std::sync::Arc;

use rustc_ast::expand::StrippedCfgItem;
use rustc_ast::{self as ast, Crate, NodeId, attr};
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::StabilityLevel;
use rustc_data_structures::intern::Interned;
use rustc_errors::{Applicability, DiagCtxtHandle, StashKey};
use rustc_expand::base::{
    Annotatable, DeriveResolution, Indeterminate, ResolverExpand, SyntaxExtension,
    SyntaxExtensionKind,
};
use rustc_expand::compile_declarative_macro;
use rustc_expand::expand::{
    AstFragment, AstFragmentKind, Invocation, InvocationKind, SupportsMacroExpansion,
};
use rustc_hir::def::{self, DefKind, Namespace, NonMacroAttrKind};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId};
use rustc_middle::middle::stability;
use rustc_middle::ty::{RegisteredTools, TyCtxt, Visibility};
use rustc_session::lint::BuiltinLintDiag;
use rustc_session::lint::builtin::{
    LEGACY_DERIVE_HELPERS, OUT_OF_SCOPE_MACRO_CALLS, UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
    UNUSED_MACRO_RULES, UNUSED_MACROS,
};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::edition::Edition;
use rustc_span::hygiene::{self, AstPass, ExpnData, ExpnKind, LocalExpnId, MacroKind};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};

use crate::Namespace::*;
use crate::errors::{
    self, AddAsNonDerive, CannotDetermineMacroResolution, CannotFindIdentInThisScope,
    MacroExpectedFound, RemoveSurroundingDerive,
};
use crate::imports::Import;
use crate::{
    BindingKey, DeriveData, Determinacy, Finalize, InvocationParent, MacroData, ModuleKind,
    ModuleOrUniformRoot, NameBinding, NameBindingKind, ParentScope, PathResult, ResolutionError,
    Resolver, ScopeSet, Segment, ToNameBinding, Used,
};

type Res = def::Res<NodeId>;

/// Binding produced by a `macro_rules` item.
/// Not modularized, can shadow previous `macro_rules` bindings, etc.
#[derive(Debug)]
pub(crate) struct MacroRulesBinding<'ra> {
    pub(crate) binding: NameBinding<'ra>,
    /// `macro_rules` scope into which the `macro_rules` item was planted.
    pub(crate) parent_macro_rules_scope: MacroRulesScopeRef<'ra>,
    pub(crate) ident: Ident,
}

/// The scope introduced by a `macro_rules!` macro.
/// This starts at the macro's definition and ends at the end of the macro's parent
/// module (named or unnamed), or even further if it escapes with `#[macro_use]`.
/// Some macro invocations need to introduce `macro_rules` scopes too because they
/// can potentially expand into macro definitions.
#[derive(Copy, Clone, Debug)]
pub(crate) enum MacroRulesScope<'ra> {
    /// Empty "root" scope at the crate start containing no names.
    Empty,
    /// The scope introduced by a `macro_rules!` macro definition.
    Binding(&'ra MacroRulesBinding<'ra>),
    /// The scope introduced by a macro invocation that can potentially
    /// create a `macro_rules!` macro definition.
    Invocation(LocalExpnId),
}

/// `macro_rules!` scopes are always kept by reference and inside a cell.
/// The reason is that we update scopes with value `MacroRulesScope::Invocation(invoc_id)`
/// in-place after `invoc_id` gets expanded.
/// This helps to avoid uncontrollable growth of `macro_rules!` scope chains,
/// which usually grow linearly with the number of macro invocations
/// in a module (including derives) and hurt performance.
pub(crate) type MacroRulesScopeRef<'ra> = Interned<'ra, Cell<MacroRulesScope<'ra>>>;

/// Macro namespace is separated into two sub-namespaces, one for bang macros and
/// one for attribute-like macros (attributes, derives).
/// We ignore resolutions from one sub-namespace when searching names in scope for another.
pub(crate) fn sub_namespace_match(
    candidate: Option<MacroKind>,
    requirement: Option<MacroKind>,
) -> bool {
    #[derive(PartialEq)]
    enum SubNS {
        Bang,
        AttrLike,
    }
    let sub_ns = |kind| match kind {
        MacroKind::Bang => SubNS::Bang,
        MacroKind::Attr | MacroKind::Derive => SubNS::AttrLike,
    };
    let candidate = candidate.map(sub_ns);
    let requirement = requirement.map(sub_ns);
    // "No specific sub-namespace" means "matches anything" for both requirements and candidates.
    candidate.is_none() || requirement.is_none() || candidate == requirement
}

// We don't want to format a path using pretty-printing,
// `format!("{}", path)`, because that tries to insert
// line-breaks and is slow.
fn fast_print_path(path: &ast::Path) -> Symbol {
    if let [segment] = path.segments.as_slice() {
        segment.ident.name
    } else {
        let mut path_str = String::with_capacity(64);
        for (i, segment) in path.segments.iter().enumerate() {
            if i != 0 {
                path_str.push_str("::");
            }
            if segment.ident.name != kw::PathRoot {
                path_str.push_str(segment.ident.as_str())
            }
        }
        Symbol::intern(&path_str)
    }
}

pub(crate) fn registered_tools(tcx: TyCtxt<'_>, (): ()) -> RegisteredTools {
    let (_, pre_configured_attrs) = &*tcx.crate_for_resolver(()).borrow();
    registered_tools_ast(tcx.dcx(), pre_configured_attrs)
}

pub fn registered_tools_ast(
    dcx: DiagCtxtHandle<'_>,
    pre_configured_attrs: &[ast::Attribute],
) -> RegisteredTools {
    let mut registered_tools = RegisteredTools::default();
    for attr in attr::filter_by_name(pre_configured_attrs, sym::register_tool) {
        for meta_item_inner in attr.meta_item_list().unwrap_or_default() {
            match meta_item_inner.ident() {
                Some(ident) => {
                    if let Some(old_ident) = registered_tools.replace(ident) {
                        dcx.emit_err(errors::ToolWasAlreadyRegistered {
                            span: ident.span,
                            tool: ident,
                            old_ident_span: old_ident.span,
                        });
                    }
                }
                None => {
                    dcx.emit_err(errors::ToolOnlyAcceptsIdentifiers {
                        span: meta_item_inner.span(),
                        tool: sym::register_tool,
                    });
                }
            }
        }
    }
    // We implicitly add `rustfmt`, `clippy`, `diagnostic`, `miri` and `rust_analyzer` to known
    // tools, but it's not an error to register them explicitly.
    let predefined_tools =
        [sym::clippy, sym::rustfmt, sym::diagnostic, sym::miri, sym::rust_analyzer];
    registered_tools.extend(predefined_tools.iter().cloned().map(Ident::with_dummy_span));
    registered_tools
}

impl<'ra, 'tcx> ResolverExpand for Resolver<'ra, 'tcx> {
    fn next_node_id(&mut self) -> NodeId {
        self.next_node_id()
    }

    fn invocation_parent(&self, id: LocalExpnId) -> LocalDefId {
        self.invocation_parents[&id].parent_def
    }

    fn resolve_dollar_crates(&mut self) {
        hygiene::update_dollar_crate_names(|ctxt| {
            let ident = Ident::new(kw::DollarCrate, DUMMY_SP.with_ctxt(ctxt));
            match self.resolve_crate_root(ident).kind {
                ModuleKind::Def(.., name) if let Some(name) = name => name,
                _ => kw::Crate,
            }
        });
    }

    fn visit_ast_fragment_with_placeholders(
        &mut self,
        expansion: LocalExpnId,
        fragment: &AstFragment,
    ) {
        // Integrate the new AST fragment into all the definition and module structures.
        // We are inside the `expansion` now, but other parent scope components are still the same.
        let parent_scope = ParentScope { expansion, ..self.invocation_parent_scopes[&expansion] };
        let output_macro_rules_scope = self.build_reduced_graph(fragment, parent_scope);
        self.output_macro_rules_scopes.insert(expansion, output_macro_rules_scope);

        parent_scope.module.unexpanded_invocations.borrow_mut().remove(&expansion);
        if let Some(unexpanded_invocations) =
            self.impl_unexpanded_invocations.get_mut(&self.invocation_parent(expansion))
        {
            unexpanded_invocations.remove(&expansion);
        }
    }

    fn register_builtin_macro(&mut self, name: Symbol, ext: SyntaxExtensionKind) {
        if self.builtin_macros.insert(name, ext).is_some() {
            self.dcx().bug(format!("built-in macro `{name}` was already registered"));
        }
    }

    // Create a new Expansion with a definition site of the provided module, or
    // a fake empty `#[no_implicit_prelude]` module if no module is provided.
    fn expansion_for_ast_pass(
        &mut self,
        call_site: Span,
        pass: AstPass,
        features: &[Symbol],
        parent_module_id: Option<NodeId>,
    ) -> LocalExpnId {
        let parent_module =
            parent_module_id.map(|module_id| self.local_def_id(module_id).to_def_id());
        let expn_id = LocalExpnId::fresh(
            ExpnData::allow_unstable(
                ExpnKind::AstPass(pass),
                call_site,
                self.tcx.sess.edition(),
                features.into(),
                None,
                parent_module,
            ),
            self.create_stable_hashing_context(),
        );

        let parent_scope =
            parent_module.map_or(self.empty_module, |def_id| self.expect_module(def_id));
        self.ast_transform_scopes.insert(expn_id, parent_scope);

        expn_id
    }

    fn resolve_imports(&mut self) {
        self.resolve_imports()
    }

    fn resolve_macro_invocation(
        &mut self,
        invoc: &Invocation,
        eager_expansion_root: LocalExpnId,
        force: bool,
    ) -> Result<Arc<SyntaxExtension>, Indeterminate> {
        let invoc_id = invoc.expansion_data.id;
        let parent_scope = match self.invocation_parent_scopes.get(&invoc_id) {
            Some(parent_scope) => *parent_scope,
            None => {
                // If there's no entry in the table, then we are resolving an eagerly expanded
                // macro, which should inherit its parent scope from its eager expansion root -
                // the macro that requested this eager expansion.
                let parent_scope = *self
                    .invocation_parent_scopes
                    .get(&eager_expansion_root)
                    .expect("non-eager expansion without a parent scope");
                self.invocation_parent_scopes.insert(invoc_id, parent_scope);
                parent_scope
            }
        };

        let (mut derives, mut inner_attr, mut deleg_impl) = (&[][..], false, None);
        let (path, kind) = match invoc.kind {
            InvocationKind::Attr { ref attr, derives: ref attr_derives, .. } => {
                derives = self.arenas.alloc_ast_paths(attr_derives);
                inner_attr = attr.style == ast::AttrStyle::Inner;
                (&attr.get_normal_item().path, MacroKind::Attr)
            }
            InvocationKind::Bang { ref mac, .. } => (&mac.path, MacroKind::Bang),
            InvocationKind::Derive { ref path, .. } => (path, MacroKind::Derive),
            InvocationKind::GlobDelegation { ref item, .. } => {
                let ast::AssocItemKind::DelegationMac(deleg) = &item.kind else { unreachable!() };
                deleg_impl = Some(self.invocation_parent(invoc_id));
                // It is sufficient to consider glob delegation a bang macro for now.
                (&deleg.prefix, MacroKind::Bang)
            }
        };

        // Derives are not included when `invocations` are collected, so we have to add them here.
        let parent_scope = &ParentScope { derives, ..parent_scope };
        let supports_macro_expansion = invoc.fragment_kind.supports_macro_expansion();
        let node_id = invoc.expansion_data.lint_node_id;
        // This is a heuristic, but it's good enough for the lint.
        let looks_like_invoc_in_mod_inert_attr = self
            .invocation_parents
            .get(&invoc_id)
            .or_else(|| self.invocation_parents.get(&eager_expansion_root))
            .filter(|&&InvocationParent { parent_def: mod_def_id, in_attr, .. }| {
                in_attr
                    && invoc.fragment_kind == AstFragmentKind::Expr
                    && self.tcx.def_kind(mod_def_id) == DefKind::Mod
            })
            .map(|&InvocationParent { parent_def: mod_def_id, .. }| mod_def_id);
        let sugg_span = match &invoc.kind {
            InvocationKind::Attr { item: Annotatable::Item(item), .. }
                if !item.span.from_expansion() =>
            {
                Some(item.span.shrink_to_lo())
            }
            _ => None,
        };
        let (ext, res) = self.smart_resolve_macro_path(
            path,
            kind,
            supports_macro_expansion,
            inner_attr,
            parent_scope,
            node_id,
            force,
            deleg_impl,
            looks_like_invoc_in_mod_inert_attr,
            sugg_span,
        )?;

        let span = invoc.span();
        let def_id = if deleg_impl.is_some() { None } else { res.opt_def_id() };
        invoc_id.set_expn_data(
            ext.expn_data(
                parent_scope.expansion,
                span,
                fast_print_path(path),
                def_id,
                def_id.map(|def_id| self.macro_def_scope(def_id).nearest_parent_mod()),
            ),
            self.create_stable_hashing_context(),
        );

        Ok(ext)
    }

    fn record_macro_rule_usage(&mut self, id: NodeId, rule_i: usize) {
        if let Some(rules) = self.unused_macro_rules.get_mut(&id) {
            rules.remove(&rule_i);
        }
    }

    fn check_unused_macros(&mut self) {
        for (_, &(node_id, ident)) in self.unused_macros.iter() {
            self.lint_buffer.buffer_lint(
                UNUSED_MACROS,
                node_id,
                ident.span,
                BuiltinLintDiag::UnusedMacroDefinition(ident.name),
            );
            // Do not report unused individual rules if the entire macro is unused
            self.unused_macro_rules.swap_remove(&node_id);
        }

        for (&node_id, unused_arms) in self.unused_macro_rules.iter() {
            for (&arm_i, &(ident, rule_span)) in unused_arms.to_sorted_stable_ord() {
                self.lint_buffer.buffer_lint(
                    UNUSED_MACRO_RULES,
                    node_id,
                    rule_span,
                    BuiltinLintDiag::MacroRuleNeverUsed(arm_i, ident.name),
                );
            }
        }
    }

    fn has_derive_copy(&self, expn_id: LocalExpnId) -> bool {
        self.containers_deriving_copy.contains(&expn_id)
    }

    fn resolve_derives(
        &mut self,
        expn_id: LocalExpnId,
        force: bool,
        derive_paths: &dyn Fn() -> Vec<DeriveResolution>,
    ) -> Result<(), Indeterminate> {
        // Block expansion of the container until we resolve all derives in it.
        // This is required for two reasons:
        // - Derive helper attributes are in scope for the item to which the `#[derive]`
        //   is applied, so they have to be produced by the container's expansion rather
        //   than by individual derives.
        // - Derives in the container need to know whether one of them is a built-in `Copy`.
        // Temporarily take the data to avoid borrow checker conflicts.
        let mut derive_data = mem::take(&mut self.derive_data);
        let entry = derive_data.entry(expn_id).or_insert_with(|| DeriveData {
            resolutions: derive_paths(),
            helper_attrs: Vec::new(),
            has_derive_copy: false,
        });
        let parent_scope = self.invocation_parent_scopes[&expn_id];
        for (i, resolution) in entry.resolutions.iter_mut().enumerate() {
            if resolution.exts.is_none() {
                resolution.exts = Some(
                    match self.resolve_macro_path(
                        &resolution.path,
                        Some(MacroKind::Derive),
                        &parent_scope,
                        true,
                        force,
                        None,
                        None,
                    ) {
                        Ok((Some(ext), _)) => {
                            if !ext.helper_attrs.is_empty() {
                                let last_seg = resolution.path.segments.last().unwrap();
                                let span = last_seg.ident.span.normalize_to_macros_2_0();
                                entry.helper_attrs.extend(
                                    ext.helper_attrs
                                        .iter()
                                        .map(|name| (i, Ident::new(*name, span))),
                                );
                            }
                            entry.has_derive_copy |= ext.builtin_name == Some(sym::Copy);
                            ext
                        }
                        Ok(_) | Err(Determinacy::Determined) => self.dummy_ext(MacroKind::Derive),
                        Err(Determinacy::Undetermined) => {
                            assert!(self.derive_data.is_empty());
                            self.derive_data = derive_data;
                            return Err(Indeterminate);
                        }
                    },
                );
            }
        }
        // Sort helpers in a stable way independent from the derive resolution order.
        entry.helper_attrs.sort_by_key(|(i, _)| *i);
        let helper_attrs = entry
            .helper_attrs
            .iter()
            .map(|(_, ident)| {
                let res = Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);
                let binding = (res, Visibility::<DefId>::Public, ident.span, expn_id)
                    .to_name_binding(self.arenas);
                (*ident, binding)
            })
            .collect();
        self.helper_attrs.insert(expn_id, helper_attrs);
        // Mark this derive as having `Copy` either if it has `Copy` itself or if its parent derive
        // has `Copy`, to support cases like `#[derive(Clone, Copy)] #[derive(Debug)]`.
        if entry.has_derive_copy || self.has_derive_copy(parent_scope.expansion) {
            self.containers_deriving_copy.insert(expn_id);
        }
        assert!(self.derive_data.is_empty());
        self.derive_data = derive_data;
        Ok(())
    }

    fn take_derive_resolutions(&mut self, expn_id: LocalExpnId) -> Option<Vec<DeriveResolution>> {
        self.derive_data.remove(&expn_id).map(|data| data.resolutions)
    }

    // The function that implements the resolution logic of `#[cfg_accessible(path)]`.
    // Returns true if the path can certainly be resolved in one of three namespaces,
    // returns false if the path certainly cannot be resolved in any of the three namespaces.
    // Returns `Indeterminate` if we cannot give a certain answer yet.
    fn cfg_accessible(
        &mut self,
        expn_id: LocalExpnId,
        path: &ast::Path,
    ) -> Result<bool, Indeterminate> {
        self.path_accessible(expn_id, path, &[TypeNS, ValueNS, MacroNS])
    }

    fn macro_accessible(
        &mut self,
        expn_id: LocalExpnId,
        path: &ast::Path,
    ) -> Result<bool, Indeterminate> {
        self.path_accessible(expn_id, path, &[MacroNS])
    }

    fn get_proc_macro_quoted_span(&self, krate: CrateNum, id: usize) -> Span {
        self.cstore().get_proc_macro_quoted_span_untracked(krate, id, self.tcx.sess)
    }

    fn declare_proc_macro(&mut self, id: NodeId) {
        self.proc_macros.push(self.local_def_id(id))
    }

    fn append_stripped_cfg_item(&mut self, parent_node: NodeId, ident: Ident, cfg: ast::MetaItem) {
        self.stripped_cfg_items.push(StrippedCfgItem { parent_module: parent_node, ident, cfg });
    }

    fn registered_tools(&self) -> &RegisteredTools {
        self.registered_tools
    }

    fn register_glob_delegation(&mut self, invoc_id: LocalExpnId) {
        self.glob_delegation_invoc_ids.insert(invoc_id);
    }

    fn glob_delegation_suffixes(
        &mut self,
        trait_def_id: DefId,
        impl_def_id: LocalDefId,
    ) -> Result<Vec<(Ident, Option<Ident>)>, Indeterminate> {
        let target_trait = self.expect_module(trait_def_id);
        if !target_trait.unexpanded_invocations.borrow().is_empty() {
            return Err(Indeterminate);
        }
        // FIXME: Instead of waiting try generating all trait methods, and pruning
        // the shadowed ones a bit later, e.g. when all macro expansion completes.
        // Pros: expansion will be stuck less (but only in exotic cases), the implementation may be
        // less hacky.
        // Cons: More code is generated just to be deleted later, deleting already created `DefId`s
        // may be nontrivial.
        if let Some(unexpanded_invocations) = self.impl_unexpanded_invocations.get(&impl_def_id)
            && !unexpanded_invocations.is_empty()
        {
            return Err(Indeterminate);
        }

        let mut idents = Vec::new();
        target_trait.for_each_child(self, |this, ident, ns, _binding| {
            // FIXME: Adjust hygiene for idents from globs, like for glob imports.
            if let Some(overriding_keys) = this.impl_binding_keys.get(&impl_def_id)
                && overriding_keys.contains(&BindingKey::new(ident.normalize_to_macros_2_0(), ns))
            {
                // The name is overridden, do not produce it from the glob delegation.
            } else {
                idents.push((ident, None));
            }
        });
        Ok(idents)
    }

    fn insert_impl_trait_name(&mut self, id: NodeId, name: Symbol) {
        self.impl_trait_names.insert(id, name);
    }
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    /// Resolve macro path with error reporting and recovery.
    /// Uses dummy syntax extensions for unresolved macros or macros with unexpected resolutions
    /// for better error recovery.
    fn smart_resolve_macro_path(
        &mut self,
        path: &ast::Path,
        kind: MacroKind,
        supports_macro_expansion: SupportsMacroExpansion,
        inner_attr: bool,
        parent_scope: &ParentScope<'ra>,
        node_id: NodeId,
        force: bool,
        deleg_impl: Option<LocalDefId>,
        invoc_in_mod_inert_attr: Option<LocalDefId>,
        suggestion_span: Option<Span>,
    ) -> Result<(Arc<SyntaxExtension>, Res), Indeterminate> {
        let (ext, res) = match self.resolve_macro_or_delegation_path(
            path,
            Some(kind),
            parent_scope,
            true,
            force,
            deleg_impl,
            invoc_in_mod_inert_attr.map(|def_id| (def_id, node_id)),
            None,
            suggestion_span,
        ) {
            Ok((Some(ext), res)) => (ext, res),
            Ok((None, res)) => (self.dummy_ext(kind), res),
            Err(Determinacy::Determined) => (self.dummy_ext(kind), Res::Err),
            Err(Determinacy::Undetermined) => return Err(Indeterminate),
        };

        // Everything below is irrelevant to glob delegation, take a shortcut.
        if deleg_impl.is_some() {
            if !matches!(res, Res::Err | Res::Def(DefKind::Trait, _)) {
                self.dcx().emit_err(MacroExpectedFound {
                    span: path.span,
                    expected: "trait",
                    article: "a",
                    found: res.descr(),
                    macro_path: &pprust::path_to_string(path),
                    remove_surrounding_derive: None,
                    add_as_non_derive: None,
                });
                return Ok((self.dummy_ext(kind), Res::Err));
            }

            return Ok((ext, res));
        }

        // Report errors for the resolved macro.
        for segment in &path.segments {
            if let Some(args) = &segment.args {
                self.dcx().emit_err(errors::GenericArgumentsInMacroPath { span: args.span() });
            }
            if kind == MacroKind::Attr && segment.ident.as_str().starts_with("rustc") {
                self.dcx().emit_err(errors::AttributesStartingWithRustcAreReserved {
                    span: segment.ident.span,
                });
            }
        }

        match res {
            Res::Def(DefKind::Macro(_), def_id) => {
                if let Some(def_id) = def_id.as_local() {
                    self.unused_macros.swap_remove(&def_id);
                    if self.proc_macro_stubs.contains(&def_id) {
                        self.dcx().emit_err(errors::ProcMacroSameCrate {
                            span: path.span,
                            is_test: self.tcx.sess.is_test_crate(),
                        });
                    }
                }
            }
            Res::NonMacroAttr(..) | Res::Err => {}
            _ => panic!("expected `DefKind::Macro` or `Res::NonMacroAttr`"),
        };

        self.check_stability_and_deprecation(&ext, path, node_id);

        let unexpected_res = if ext.macro_kind() != kind {
            Some((kind.article(), kind.descr_expected()))
        } else if matches!(res, Res::Def(..)) {
            match supports_macro_expansion {
                SupportsMacroExpansion::No => Some(("a", "non-macro attribute")),
                SupportsMacroExpansion::Yes { supports_inner_attrs } => {
                    if inner_attr && !supports_inner_attrs {
                        Some(("a", "non-macro inner attribute"))
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        };
        if let Some((article, expected)) = unexpected_res {
            let path_str = pprust::path_to_string(path);

            let mut err = MacroExpectedFound {
                span: path.span,
                expected,
                article,
                found: res.descr(),
                macro_path: &path_str,
                remove_surrounding_derive: None,
                add_as_non_derive: None,
            };

            // Suggest moving the macro out of the derive() if the macro isn't Derive
            if !path.span.from_expansion()
                && kind == MacroKind::Derive
                && ext.macro_kind() != MacroKind::Derive
            {
                err.remove_surrounding_derive = Some(RemoveSurroundingDerive { span: path.span });
                err.add_as_non_derive = Some(AddAsNonDerive { macro_path: &path_str });
            }

            self.dcx().emit_err(err);

            return Ok((self.dummy_ext(kind), Res::Err));
        }

        // We are trying to avoid reporting this error if other related errors were reported.
        if res != Res::Err && inner_attr && !self.tcx.features().custom_inner_attributes() {
            let is_macro = match res {
                Res::Def(..) => true,
                Res::NonMacroAttr(..) => false,
                _ => unreachable!(),
            };
            let msg = if is_macro {
                "inner macro attributes are unstable"
            } else {
                "custom inner attributes are unstable"
            };
            feature_err(&self.tcx.sess, sym::custom_inner_attributes, path.span, msg).emit();
        }

        if res == Res::NonMacroAttr(NonMacroAttrKind::Tool)
            && let [namespace, attribute, ..] = &*path.segments
            && namespace.ident.name == sym::diagnostic
            && ![sym::on_unimplemented, sym::do_not_recommend].contains(&attribute.ident.name)
        {
            let typo_name = find_best_match_for_name(
                &[sym::on_unimplemented, sym::do_not_recommend],
                attribute.ident.name,
                Some(5),
            );

            self.tcx.sess.psess.buffer_lint(
                UNKNOWN_OR_MALFORMED_DIAGNOSTIC_ATTRIBUTES,
                attribute.span(),
                node_id,
                BuiltinLintDiag::UnknownDiagnosticAttribute { span: attribute.span(), typo_name },
            );
        }

        Ok((ext, res))
    }

    pub(crate) fn resolve_macro_path(
        &mut self,
        path: &ast::Path,
        kind: Option<MacroKind>,
        parent_scope: &ParentScope<'ra>,
        trace: bool,
        force: bool,
        ignore_import: Option<Import<'ra>>,
        suggestion_span: Option<Span>,
    ) -> Result<(Option<Arc<SyntaxExtension>>, Res), Determinacy> {
        self.resolve_macro_or_delegation_path(
            path,
            kind,
            parent_scope,
            trace,
            force,
            None,
            None,
            ignore_import,
            suggestion_span,
        )
    }

    fn resolve_macro_or_delegation_path(
        &mut self,
        ast_path: &ast::Path,
        kind: Option<MacroKind>,
        parent_scope: &ParentScope<'ra>,
        trace: bool,
        force: bool,
        deleg_impl: Option<LocalDefId>,
        invoc_in_mod_inert_attr: Option<(LocalDefId, NodeId)>,
        ignore_import: Option<Import<'ra>>,
        suggestion_span: Option<Span>,
    ) -> Result<(Option<Arc<SyntaxExtension>>, Res), Determinacy> {
        let path_span = ast_path.span;
        let mut path = Segment::from_path(ast_path);

        // Possibly apply the macro helper hack
        if deleg_impl.is_none()
            && kind == Some(MacroKind::Bang)
            && let [segment] = path.as_slice()
            && segment.ident.span.ctxt().outer_expn_data().local_inner_macros
        {
            let root = Ident::new(kw::DollarCrate, segment.ident.span);
            path.insert(0, Segment::from_ident(root));
        }

        let res = if deleg_impl.is_some() || path.len() > 1 {
            let ns = if deleg_impl.is_some() { TypeNS } else { MacroNS };
            let res = match self.maybe_resolve_path(&path, Some(ns), parent_scope, ignore_import) {
                PathResult::NonModule(path_res) if let Some(res) = path_res.full_res() => Ok(res),
                PathResult::Indeterminate if !force => return Err(Determinacy::Undetermined),
                PathResult::NonModule(..)
                | PathResult::Indeterminate
                | PathResult::Failed { .. } => Err(Determinacy::Determined),
                PathResult::Module(ModuleOrUniformRoot::Module(module)) => {
                    Ok(module.res().unwrap())
                }
                PathResult::Module(..) => unreachable!(),
            };

            if trace {
                let kind = kind.expect("macro kind must be specified if tracing is enabled");
                self.multi_segment_macro_resolutions.push((
                    path,
                    path_span,
                    kind,
                    *parent_scope,
                    res.ok(),
                    ns,
                ));
            }

            self.prohibit_imported_non_macro_attrs(None, res.ok(), path_span);
            res
        } else {
            let scope_set = kind.map_or(ScopeSet::All(MacroNS), ScopeSet::Macro);
            let binding = self.early_resolve_ident_in_lexical_scope(
                path[0].ident,
                scope_set,
                parent_scope,
                None,
                force,
                None,
                None,
            );
            if let Err(Determinacy::Undetermined) = binding {
                return Err(Determinacy::Undetermined);
            }

            if trace {
                let kind = kind.expect("macro kind must be specified if tracing is enabled");
                self.single_segment_macro_resolutions.push((
                    path[0].ident,
                    kind,
                    *parent_scope,
                    binding.ok(),
                    suggestion_span,
                ));
            }

            let res = binding.map(|binding| binding.res());
            self.prohibit_imported_non_macro_attrs(binding.ok(), res.ok(), path_span);
            self.report_out_of_scope_macro_calls(
                ast_path,
                parent_scope,
                invoc_in_mod_inert_attr,
                binding.ok(),
            );
            res
        };

        let res = res?;
        let ext = match deleg_impl {
            Some(impl_def_id) => match res {
                def::Res::Def(DefKind::Trait, def_id) => {
                    let edition = self.tcx.sess.edition();
                    Some(Arc::new(SyntaxExtension::glob_delegation(def_id, impl_def_id, edition)))
                }
                _ => None,
            },
            None => self.get_macro(res).map(|macro_data| Arc::clone(&macro_data.ext)),
        };
        Ok((ext, res))
    }

    pub(crate) fn finalize_macro_resolutions(&mut self, krate: &Crate) {
        let check_consistency = |this: &mut Self,
                                 path: &[Segment],
                                 span,
                                 kind: MacroKind,
                                 initial_res: Option<Res>,
                                 res: Res| {
            if let Some(initial_res) = initial_res {
                if res != initial_res {
                    // Make sure compilation does not succeed if preferred macro resolution
                    // has changed after the macro had been expanded. In theory all such
                    // situations should be reported as errors, so this is a bug.
                    this.dcx().span_delayed_bug(span, "inconsistent resolution for a macro");
                }
            } else if this.tcx.dcx().has_errors().is_none() && this.privacy_errors.is_empty() {
                // It's possible that the macro was unresolved (indeterminate) and silently
                // expanded into a dummy fragment for recovery during expansion.
                // Now, post-expansion, the resolution may succeed, but we can't change the
                // past and need to report an error.
                // However, non-speculative `resolve_path` can successfully return private items
                // even if speculative `resolve_path` returned nothing previously, so we skip this
                // less informative error if no other error is reported elsewhere.

                let err = this.dcx().create_err(CannotDetermineMacroResolution {
                    span,
                    kind: kind.descr(),
                    path: Segment::names_to_string(path),
                });
                err.stash(span, StashKey::UndeterminedMacroResolution);
            }
        };

        let macro_resolutions = mem::take(&mut self.multi_segment_macro_resolutions);
        for (mut path, path_span, kind, parent_scope, initial_res, ns) in macro_resolutions {
            // FIXME: Path resolution will ICE if segment IDs present.
            for seg in &mut path {
                seg.id = None;
            }
            match self.resolve_path(
                &path,
                Some(ns),
                &parent_scope,
                Some(Finalize::new(ast::CRATE_NODE_ID, path_span)),
                None,
                None,
            ) {
                PathResult::NonModule(path_res) if let Some(res) = path_res.full_res() => {
                    check_consistency(self, &path, path_span, kind, initial_res, res)
                }
                // This may be a trait for glob delegation expansions.
                PathResult::Module(ModuleOrUniformRoot::Module(module)) => check_consistency(
                    self,
                    &path,
                    path_span,
                    kind,
                    initial_res,
                    module.res().unwrap(),
                ),
                path_res @ (PathResult::NonModule(..) | PathResult::Failed { .. }) => {
                    let mut suggestion = None;
                    let (span, label, module, segment) =
                        if let PathResult::Failed { span, label, module, segment_name, .. } =
                            path_res
                        {
                            // try to suggest if it's not a macro, maybe a function
                            if let PathResult::NonModule(partial_res) =
                                self.maybe_resolve_path(&path, Some(ValueNS), &parent_scope, None)
                                && partial_res.unresolved_segments() == 0
                            {
                                let sm = self.tcx.sess.source_map();
                                let exclamation_span = sm.next_point(span);
                                suggestion = Some((
                                    vec![(exclamation_span, "".to_string())],
                                    format!(
                                        "{} is not a macro, but a {}, try to remove `!`",
                                        Segment::names_to_string(&path),
                                        partial_res.base_res().descr()
                                    ),
                                    Applicability::MaybeIncorrect,
                                ));
                            }
                            (span, label, module, segment_name)
                        } else {
                            (
                                path_span,
                                format!(
                                    "partially resolved path in {} {}",
                                    kind.article(),
                                    kind.descr()
                                ),
                                None,
                                path.last().map(|segment| segment.ident.name).unwrap(),
                            )
                        };
                    self.report_error(
                        span,
                        ResolutionError::FailedToResolve {
                            segment: Some(segment),
                            label,
                            suggestion,
                            module,
                        },
                    );
                }
                PathResult::Module(..) | PathResult::Indeterminate => unreachable!(),
            }
        }

        let macro_resolutions = mem::take(&mut self.single_segment_macro_resolutions);
        for (ident, kind, parent_scope, initial_binding, sugg_span) in macro_resolutions {
            match self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::Macro(kind),
                &parent_scope,
                Some(Finalize::new(ast::CRATE_NODE_ID, ident.span)),
                true,
                None,
                None,
            ) {
                Ok(binding) => {
                    let initial_res = initial_binding.map(|initial_binding| {
                        self.record_use(ident, initial_binding, Used::Other);
                        initial_binding.res()
                    });
                    let res = binding.res();
                    let seg = Segment::from_ident(ident);
                    check_consistency(self, &[seg], ident.span, kind, initial_res, res);
                    if res == Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat) {
                        let node_id = self
                            .invocation_parents
                            .get(&parent_scope.expansion)
                            .map_or(ast::CRATE_NODE_ID, |parent| {
                                self.def_id_to_node_id(parent.parent_def)
                            });
                        self.lint_buffer.buffer_lint(
                            LEGACY_DERIVE_HELPERS,
                            node_id,
                            ident.span,
                            BuiltinLintDiag::LegacyDeriveHelpers(binding.span),
                        );
                    }
                }
                Err(..) => {
                    let expected = kind.descr_expected();

                    let mut err = self.dcx().create_err(CannotFindIdentInThisScope {
                        span: ident.span,
                        expected,
                        ident,
                    });
                    self.unresolved_macro_suggestions(
                        &mut err,
                        kind,
                        &parent_scope,
                        ident,
                        krate,
                        sugg_span,
                    );
                    err.emit();
                }
            }
        }

        let builtin_attrs = mem::take(&mut self.builtin_attrs);
        for (ident, parent_scope) in builtin_attrs {
            let _ = self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::Macro(MacroKind::Attr),
                &parent_scope,
                Some(Finalize::new(ast::CRATE_NODE_ID, ident.span)),
                true,
                None,
                None,
            );
        }
    }

    fn check_stability_and_deprecation(
        &mut self,
        ext: &SyntaxExtension,
        path: &ast::Path,
        node_id: NodeId,
    ) {
        let span = path.span;
        if let Some(stability) = &ext.stability {
            if let StabilityLevel::Unstable { reason, issue, is_soft, implied_by, .. } =
                stability.level
            {
                let feature = stability.feature;

                let is_allowed =
                    |feature| self.tcx.features().enabled(feature) || span.allows_unstable(feature);
                let allowed_by_implication = implied_by.is_some_and(|feature| is_allowed(feature));
                if !is_allowed(feature) && !allowed_by_implication {
                    let lint_buffer = &mut self.lint_buffer;
                    let soft_handler = |lint, span, msg: String| {
                        lint_buffer.buffer_lint(
                            lint,
                            node_id,
                            span,
                            BuiltinLintDiag::UnstableFeature(
                                // FIXME make this translatable
                                msg.into(),
                            ),
                        )
                    };
                    stability::report_unstable(
                        self.tcx.sess,
                        feature,
                        reason.to_opt_reason(),
                        issue,
                        None,
                        is_soft,
                        span,
                        soft_handler,
                        stability::UnstableKind::Regular,
                    );
                }
            }
        }
        if let Some(depr) = &ext.deprecation {
            let path = pprust::path_to_string(path);
            stability::early_report_macro_deprecation(
                &mut self.lint_buffer,
                depr,
                span,
                node_id,
                path,
            );
        }
    }

    fn prohibit_imported_non_macro_attrs(
        &self,
        binding: Option<NameBinding<'ra>>,
        res: Option<Res>,
        span: Span,
    ) {
        if let Some(Res::NonMacroAttr(kind)) = res {
            if kind != NonMacroAttrKind::Tool && binding.is_none_or(|b| b.is_import()) {
                let binding_span = binding.map(|binding| binding.span);
                self.dcx().emit_err(errors::CannotUseThroughAnImport {
                    span,
                    article: kind.article(),
                    descr: kind.descr(),
                    binding_span,
                });
            }
        }
    }

    fn report_out_of_scope_macro_calls(
        &mut self,
        path: &ast::Path,
        parent_scope: &ParentScope<'ra>,
        invoc_in_mod_inert_attr: Option<(LocalDefId, NodeId)>,
        binding: Option<NameBinding<'ra>>,
    ) {
        if let Some((mod_def_id, node_id)) = invoc_in_mod_inert_attr
            && let Some(binding) = binding
            // This is a `macro_rules` itself, not some import.
            && let NameBindingKind::Res(res) = binding.kind
            && let Res::Def(DefKind::Macro(MacroKind::Bang), def_id) = res
            // And the `macro_rules` is defined inside the attribute's module,
            // so it cannot be in scope unless imported.
            && self.tcx.is_descendant_of(def_id, mod_def_id.to_def_id())
        {
            // Try to resolve our ident ignoring `macro_rules` scopes.
            // If such resolution is successful and gives the same result
            // (e.g. if the macro is re-imported), then silence the lint.
            let no_macro_rules = self.arenas.alloc_macro_rules_scope(MacroRulesScope::Empty);
            let fallback_binding = self.early_resolve_ident_in_lexical_scope(
                path.segments[0].ident,
                ScopeSet::Macro(MacroKind::Bang),
                &ParentScope { macro_rules: no_macro_rules, ..*parent_scope },
                None,
                false,
                None,
                None,
            );
            if fallback_binding.ok().and_then(|b| b.res().opt_def_id()) != Some(def_id) {
                let location = match parent_scope.module.kind {
                    ModuleKind::Def(kind, def_id, name) => {
                        if let Some(name) = name {
                            format!("{} `{name}`", kind.descr(def_id))
                        } else {
                            "the crate root".to_string()
                        }
                    }
                    ModuleKind::Block => "this scope".to_string(),
                };
                self.tcx.sess.psess.buffer_lint(
                    OUT_OF_SCOPE_MACRO_CALLS,
                    path.span,
                    node_id,
                    BuiltinLintDiag::OutOfScopeMacroCalls {
                        span: path.span,
                        path: pprust::path_to_string(path),
                        location,
                    },
                );
            }
        }
    }

    pub(crate) fn check_reserved_macro_name(&mut self, ident: Ident, res: Res) {
        // Reserve some names that are not quite covered by the general check
        // performed on `Resolver::builtin_attrs`.
        if ident.name == sym::cfg || ident.name == sym::cfg_attr {
            let macro_kind = self.get_macro(res).map(|macro_data| macro_data.ext.macro_kind());
            if macro_kind.is_some() && sub_namespace_match(macro_kind, Some(MacroKind::Attr)) {
                self.dcx()
                    .emit_err(errors::NameReservedInAttributeNamespace { span: ident.span, ident });
            }
        }
    }

    /// Compile the macro into a `SyntaxExtension` and its rule spans.
    ///
    /// Possibly replace its expander to a pre-defined one for built-in macros.
    pub(crate) fn compile_macro(
        &mut self,
        macro_def: &ast::MacroDef,
        ident: Ident,
        attrs: &[rustc_hir::Attribute],
        span: Span,
        node_id: NodeId,
        edition: Edition,
    ) -> MacroData {
        let (mut ext, mut rule_spans) = compile_declarative_macro(
            self.tcx.sess,
            self.tcx.features(),
            macro_def,
            ident,
            attrs,
            span,
            node_id,
            edition,
        );

        if let Some(builtin_name) = ext.builtin_name {
            // The macro was marked with `#[rustc_builtin_macro]`.
            if let Some(builtin_ext_kind) = self.builtin_macros.get(&builtin_name) {
                // The macro is a built-in, replace its expander function
                // while still taking everything else from the source code.
                ext.kind = builtin_ext_kind.clone();
                rule_spans = Vec::new();
            } else {
                self.dcx().emit_err(errors::CannotFindBuiltinMacroWithName { span, ident });
            }
        }

        MacroData { ext: Arc::new(ext), rule_spans, macro_rules: macro_def.macro_rules }
    }

    fn path_accessible(
        &mut self,
        expn_id: LocalExpnId,
        path: &ast::Path,
        namespaces: &[Namespace],
    ) -> Result<bool, Indeterminate> {
        let span = path.span;
        let path = &Segment::from_path(path);
        let parent_scope = self.invocation_parent_scopes[&expn_id];

        let mut indeterminate = false;
        for ns in namespaces {
            match self.maybe_resolve_path(path, Some(*ns), &parent_scope, None) {
                PathResult::Module(ModuleOrUniformRoot::Module(_)) => return Ok(true),
                PathResult::NonModule(partial_res) if partial_res.unresolved_segments() == 0 => {
                    return Ok(true);
                }
                PathResult::NonModule(..) |
                // HACK(Urgau): This shouldn't be necessary
                PathResult::Failed { is_error_from_last_segment: false, .. } => {
                    self.dcx()
                        .emit_err(errors::CfgAccessibleUnsure { span });

                    // If we get a partially resolved NonModule in one namespace, we should get the
                    // same result in any other namespaces, so we can return early.
                    return Ok(false);
                }
                PathResult::Indeterminate => indeterminate = true,
                // We can only be sure that a path doesn't exist after having tested all the
                // possibilities, only at that time we can return false.
                PathResult::Failed { .. } => {}
                PathResult::Module(_) => panic!("unexpected path resolution"),
            }
        }

        if indeterminate {
            return Err(Indeterminate);
        }

        Ok(false)
    }
}
