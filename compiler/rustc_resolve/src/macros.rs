//! A bunch of methods and structures more or less related to resolving macros and
//! interface provided by `Resolver` to macro expander.

use crate::errors::{
    self, AddAsNonDerive, CannotFindIdentInThisScope, MacroExpectedFound, RemoveSurroundingDerive,
};
use crate::Namespace::*;
use crate::{BuiltinMacroState, Determinacy};
use crate::{DeriveData, Finalize, ParentScope, ResolutionError, Resolver, ScopeSet};
use crate::{ModuleKind, ModuleOrUniformRoot, NameBinding, PathResult, Segment};
use rustc_ast::expand::StrippedCfgItem;
use rustc_ast::{self as ast, attr, Crate, Inline, ItemKind, ModKind, NodeId};
use rustc_ast_pretty::pprust;
use rustc_attr::StabilityLevel;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{struct_span_err, Applicability};
use rustc_expand::base::{Annotatable, DeriveResolutions, Indeterminate, ResolverExpand};
use rustc_expand::base::{SyntaxExtension, SyntaxExtensionKind};
use rustc_expand::compile_declarative_macro;
use rustc_expand::expand::{AstFragment, Invocation, InvocationKind, SupportsMacroExpansion};
use rustc_hir::def::{self, DefKind, NonMacroAttrKind};
use rustc_hir::def_id::{CrateNum, LocalDefId};
use rustc_middle::middle::stability;
use rustc_middle::ty::RegisteredTools;
use rustc_middle::ty::TyCtxt;
use rustc_session::lint::builtin::{LEGACY_DERIVE_HELPERS, SOFT_UNSTABLE};
use rustc_session::lint::builtin::{UNUSED_MACROS, UNUSED_MACRO_RULES};
use rustc_session::lint::BuiltinLintDiagnostics;
use rustc_session::parse::feature_err;
use rustc_span::edition::Edition;
use rustc_span::hygiene::{self, ExpnData, ExpnKind, LocalExpnId};
use rustc_span::hygiene::{AstPass, MacroKind};
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use std::cell::Cell;
use std::mem;

type Res = def::Res<NodeId>;

/// Binding produced by a `macro_rules` item.
/// Not modularized, can shadow previous `macro_rules` bindings, etc.
#[derive(Debug)]
pub(crate) struct MacroRulesBinding<'a> {
    pub(crate) binding: NameBinding<'a>,
    /// `macro_rules` scope into which the `macro_rules` item was planted.
    pub(crate) parent_macro_rules_scope: MacroRulesScopeRef<'a>,
    pub(crate) ident: Ident,
}

/// The scope introduced by a `macro_rules!` macro.
/// This starts at the macro's definition and ends at the end of the macro's parent
/// module (named or unnamed), or even further if it escapes with `#[macro_use]`.
/// Some macro invocations need to introduce `macro_rules` scopes too because they
/// can potentially expand into macro definitions.
#[derive(Copy, Clone, Debug)]
pub(crate) enum MacroRulesScope<'a> {
    /// Empty "root" scope at the crate start containing no names.
    Empty,
    /// The scope introduced by a `macro_rules!` macro definition.
    Binding(&'a MacroRulesBinding<'a>),
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
pub(crate) type MacroRulesScopeRef<'a> = Interned<'a, Cell<MacroRulesScope<'a>>>;

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
    if path.segments.len() == 1 {
        path.segments[0].ident.name
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
    let mut registered_tools = RegisteredTools::default();
    let (_, pre_configured_attrs) = &*tcx.crate_for_resolver(()).borrow();
    for attr in attr::filter_by_name(pre_configured_attrs, sym::register_tool) {
        for nested_meta in attr.meta_item_list().unwrap_or_default() {
            match nested_meta.ident() {
                Some(ident) => {
                    if let Some(old_ident) = registered_tools.replace(ident) {
                        let msg = format!("{} `{}` was already registered", "tool", ident);
                        tcx.sess
                            .struct_span_err(ident.span, msg)
                            .span_label(old_ident.span, "already registered here")
                            .emit();
                    }
                }
                None => {
                    let msg = format!("`{}` only accepts identifiers", sym::register_tool);
                    let span = nested_meta.span();
                    tcx.sess
                        .struct_span_err(span, msg)
                        .span_label(span, "not an identifier")
                        .emit();
                }
            }
        }
    }
    // We implicitly add `rustfmt` and `clippy` to known tools,
    // but it's not an error to register them explicitly.
    let predefined_tools = [sym::clippy, sym::rustfmt];
    registered_tools.extend(predefined_tools.iter().cloned().map(Ident::with_dummy_span));
    registered_tools
}

// Some feature gates for inner attributes are reported as lints for backward compatibility.
fn soft_custom_inner_attributes_gate(path: &ast::Path, invoc: &Invocation) -> bool {
    match &path.segments[..] {
        // `#![test]`
        [seg] if seg.ident.name == sym::test => return true,
        // `#![rustfmt::skip]` on out-of-line modules
        [seg1, seg2] if seg1.ident.name == sym::rustfmt && seg2.ident.name == sym::skip => {
            if let InvocationKind::Attr { item, .. } = &invoc.kind {
                if let Annotatable::Item(item) = item {
                    if let ItemKind::Mod(_, ModKind::Loaded(_, Inline::No, _)) = item.kind {
                        return true;
                    }
                }
            }
        }
        _ => {}
    }
    false
}

impl<'a, 'tcx> ResolverExpand for Resolver<'a, 'tcx> {
    fn next_node_id(&mut self) -> NodeId {
        self.next_node_id()
    }

    fn invocation_parent(&self, id: LocalExpnId) -> LocalDefId {
        self.invocation_parents[&id].0
    }

    fn resolve_dollar_crates(&mut self) {
        hygiene::update_dollar_crate_names(|ctxt| {
            let ident = Ident::new(kw::DollarCrate, DUMMY_SP.with_ctxt(ctxt));
            match self.resolve_crate_root(ident).kind {
                ModuleKind::Def(.., name) if name != kw::Empty => name,
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
    }

    fn register_builtin_macro(&mut self, name: Symbol, ext: SyntaxExtensionKind) {
        if self.builtin_macros.insert(name, BuiltinMacroState::NotYetSeen(ext)).is_some() {
            self.tcx
                .sess
                .diagnostic()
                .bug(format!("built-in macro `{}` was already registered", name));
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
    ) -> Result<Lrc<SyntaxExtension>, Indeterminate> {
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

        let (path, kind, inner_attr, derives) = match invoc.kind {
            InvocationKind::Attr { ref attr, ref derives, .. } => (
                &attr.get_normal_item().path,
                MacroKind::Attr,
                attr.style == ast::AttrStyle::Inner,
                self.arenas.alloc_ast_paths(derives),
            ),
            InvocationKind::Bang { ref mac, .. } => (&mac.path, MacroKind::Bang, false, &[][..]),
            InvocationKind::Derive { ref path, .. } => (path, MacroKind::Derive, false, &[][..]),
        };

        // Derives are not included when `invocations` are collected, so we have to add them here.
        let parent_scope = &ParentScope { derives, ..parent_scope };
        let supports_macro_expansion = invoc.fragment_kind.supports_macro_expansion();
        let node_id = invoc.expansion_data.lint_node_id;
        let (ext, res) = self.smart_resolve_macro_path(
            path,
            kind,
            supports_macro_expansion,
            inner_attr,
            parent_scope,
            node_id,
            force,
            soft_custom_inner_attributes_gate(path, invoc),
        )?;

        let span = invoc.span();
        let def_id = res.opt_def_id();
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
        let did = self.local_def_id(id);
        self.unused_macro_rules.remove(&(did, rule_i));
    }

    fn check_unused_macros(&mut self) {
        for (_, &(node_id, ident)) in self.unused_macros.iter() {
            self.lint_buffer.buffer_lint(
                UNUSED_MACROS,
                node_id,
                ident.span,
                format!("unused macro definition: `{}`", ident.name),
            );
        }
        for (&(def_id, arm_i), &(ident, rule_span)) in self.unused_macro_rules.iter() {
            if self.unused_macros.contains_key(&def_id) {
                // We already lint the entire macro as unused
                continue;
            }
            let node_id = self.def_id_to_node_id[def_id];
            self.lint_buffer.buffer_lint(
                UNUSED_MACRO_RULES,
                node_id,
                rule_span,
                format!(
                    "{} rule of macro `{}` is never used",
                    crate::diagnostics::ordinalize(arm_i + 1),
                    ident.name
                ),
            );
        }
    }

    fn has_derive_copy(&self, expn_id: LocalExpnId) -> bool {
        self.containers_deriving_copy.contains(&expn_id)
    }

    fn resolve_derives(
        &mut self,
        expn_id: LocalExpnId,
        force: bool,
        derive_paths: &dyn Fn() -> DeriveResolutions,
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
        for (i, (path, _, opt_ext, _)) in entry.resolutions.iter_mut().enumerate() {
            if opt_ext.is_none() {
                *opt_ext = Some(
                    match self.resolve_macro_path(
                        &path,
                        Some(MacroKind::Derive),
                        &parent_scope,
                        true,
                        force,
                    ) {
                        Ok((Some(ext), _)) => {
                            if !ext.helper_attrs.is_empty() {
                                let last_seg = path.segments.last().unwrap();
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
        self.helper_attrs
            .insert(expn_id, entry.helper_attrs.iter().map(|(_, ident)| *ident).collect());
        // Mark this derive as having `Copy` either if it has `Copy` itself or if its parent derive
        // has `Copy`, to support cases like `#[derive(Clone, Copy)] #[derive(Debug)]`.
        if entry.has_derive_copy || self.has_derive_copy(parent_scope.expansion) {
            self.containers_deriving_copy.insert(expn_id);
        }
        assert!(self.derive_data.is_empty());
        self.derive_data = derive_data;
        Ok(())
    }

    fn take_derive_resolutions(&mut self, expn_id: LocalExpnId) -> Option<DeriveResolutions> {
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
        let span = path.span;
        let path = &Segment::from_path(path);
        let parent_scope = self.invocation_parent_scopes[&expn_id];

        let mut indeterminate = false;
        for ns in [TypeNS, ValueNS, MacroNS].iter().copied() {
            match self.maybe_resolve_path(path, Some(ns), &parent_scope) {
                PathResult::Module(ModuleOrUniformRoot::Module(_)) => return Ok(true),
                PathResult::NonModule(partial_res) if partial_res.unresolved_segments() == 0 => {
                    return Ok(true);
                }
                PathResult::NonModule(..) |
                // HACK(Urgau): This shouldn't be necessary
                PathResult::Failed { is_error_from_last_segment: false, .. } => {
                    self.tcx.sess
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

    fn get_proc_macro_quoted_span(&self, krate: CrateNum, id: usize) -> Span {
        self.cstore().get_proc_macro_quoted_span_untracked(krate, id, self.tcx.sess)
    }

    fn declare_proc_macro(&mut self, id: NodeId) {
        self.proc_macros.push(id)
    }

    fn append_stripped_cfg_item(&mut self, parent_node: NodeId, name: Ident, cfg: ast::MetaItem) {
        self.stripped_cfg_items.push(StrippedCfgItem { parent_module: parent_node, name, cfg });
    }

    fn registered_tools(&self) -> &RegisteredTools {
        &self.registered_tools
    }
}

impl<'a, 'tcx> Resolver<'a, 'tcx> {
    /// Resolve macro path with error reporting and recovery.
    /// Uses dummy syntax extensions for unresolved macros or macros with unexpected resolutions
    /// for better error recovery.
    fn smart_resolve_macro_path(
        &mut self,
        path: &ast::Path,
        kind: MacroKind,
        supports_macro_expansion: SupportsMacroExpansion,
        inner_attr: bool,
        parent_scope: &ParentScope<'a>,
        node_id: NodeId,
        force: bool,
        soft_custom_inner_attributes_gate: bool,
    ) -> Result<(Lrc<SyntaxExtension>, Res), Indeterminate> {
        let (ext, res) = match self.resolve_macro_path(path, Some(kind), parent_scope, true, force)
        {
            Ok((Some(ext), res)) => (ext, res),
            Ok((None, res)) => (self.dummy_ext(kind), res),
            Err(Determinacy::Determined) => (self.dummy_ext(kind), Res::Err),
            Err(Determinacy::Undetermined) => return Err(Indeterminate),
        };

        // Report errors for the resolved macro.
        for segment in &path.segments {
            if let Some(args) = &segment.args {
                self.tcx.sess.span_err(args.span(), "generic arguments in macro path");
            }
            if kind == MacroKind::Attr && segment.ident.as_str().starts_with("rustc") {
                self.tcx.sess.span_err(
                    segment.ident.span,
                    "attributes starting with `rustc` are reserved for use by the `rustc` compiler",
                );
            }
        }

        match res {
            Res::Def(DefKind::Macro(_), def_id) => {
                if let Some(def_id) = def_id.as_local() {
                    self.unused_macros.remove(&def_id);
                    if self.proc_macro_stubs.contains(&def_id) {
                        self.tcx.sess.emit_err(errors::ProcMacroSameCrate {
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
                found: res.descr(),
                macro_path: &path_str,
                ..Default::default() // Subdiagnostics default to None
            };

            // Suggest moving the macro out of the derive() if the macro isn't Derive
            if !path.span.from_expansion()
                && kind == MacroKind::Derive
                && ext.macro_kind() != MacroKind::Derive
            {
                err.remove_surrounding_derive = Some(RemoveSurroundingDerive { span: path.span });
                err.add_as_non_derive = Some(AddAsNonDerive { macro_path: &path_str });
            }

            let mut err = self.tcx.sess.create_err(err);
            err.span_label(path.span, format!("not {} {}", article, expected));

            err.emit();

            return Ok((self.dummy_ext(kind), Res::Err));
        }

        // We are trying to avoid reporting this error if other related errors were reported.
        if res != Res::Err
            && inner_attr
            && !self.tcx.sess.features_untracked().custom_inner_attributes
        {
            let msg = match res {
                Res::Def(..) => "inner macro attributes are unstable",
                Res::NonMacroAttr(..) => "custom inner attributes are unstable",
                _ => unreachable!(),
            };
            if soft_custom_inner_attributes_gate {
                self.tcx.sess.parse_sess.buffer_lint(SOFT_UNSTABLE, path.span, node_id, msg);
            } else {
                feature_err(
                    &self.tcx.sess.parse_sess,
                    sym::custom_inner_attributes,
                    path.span,
                    msg,
                )
                .emit();
            }
        }

        Ok((ext, res))
    }

    pub(crate) fn resolve_macro_path(
        &mut self,
        path: &ast::Path,
        kind: Option<MacroKind>,
        parent_scope: &ParentScope<'a>,
        trace: bool,
        force: bool,
    ) -> Result<(Option<Lrc<SyntaxExtension>>, Res), Determinacy> {
        let path_span = path.span;
        let mut path = Segment::from_path(path);

        // Possibly apply the macro helper hack
        if kind == Some(MacroKind::Bang)
            && path.len() == 1
            && path[0].ident.span.ctxt().outer_expn_data().local_inner_macros
        {
            let root = Ident::new(kw::DollarCrate, path[0].ident.span);
            path.insert(0, Segment::from_ident(root));
        }

        let res = if path.len() > 1 {
            let res = match self.maybe_resolve_path(&path, Some(MacroNS), parent_scope) {
                PathResult::NonModule(path_res) if let Some(res) = path_res.full_res() => Ok(res),
                PathResult::Indeterminate if !force => return Err(Determinacy::Undetermined),
                PathResult::NonModule(..)
                | PathResult::Indeterminate
                | PathResult::Failed { .. } => Err(Determinacy::Determined),
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
                ));
            }

            let res = binding.map(|binding| binding.res());
            self.prohibit_imported_non_macro_attrs(binding.ok(), res.ok(), path_span);
            res
        };

        res.map(|res| (self.get_macro(res).map(|macro_data| macro_data.ext), res))
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
                    this.tcx.sess.delay_span_bug(span, "inconsistent resolution for a macro");
                }
            } else {
                // It's possible that the macro was unresolved (indeterminate) and silently
                // expanded into a dummy fragment for recovery during expansion.
                // Now, post-expansion, the resolution may succeed, but we can't change the
                // past and need to report an error.
                // However, non-speculative `resolve_path` can successfully return private items
                // even if speculative `resolve_path` returned nothing previously, so we skip this
                // less informative error if the privacy error is reported elsewhere.
                if this.privacy_errors.is_empty() {
                    let msg = format!(
                        "cannot determine resolution for the {} `{}`",
                        kind.descr(),
                        Segment::names_to_string(path)
                    );
                    let msg_note = "import resolution is stuck, try simplifying macro imports";
                    this.tcx.sess.struct_span_err(span, msg).note(msg_note).emit();
                }
            }
        };

        let macro_resolutions = mem::take(&mut self.multi_segment_macro_resolutions);
        for (mut path, path_span, kind, parent_scope, initial_res) in macro_resolutions {
            // FIXME: Path resolution will ICE if segment IDs present.
            for seg in &mut path {
                seg.id = None;
            }
            match self.resolve_path(
                &path,
                Some(MacroNS),
                &parent_scope,
                Some(Finalize::new(ast::CRATE_NODE_ID, path_span)),
                None,
            ) {
                PathResult::NonModule(path_res) if let Some(res) = path_res.full_res() => {
                    check_consistency(self, &path, path_span, kind, initial_res, res)
                }
                path_res @ (PathResult::NonModule(..) | PathResult::Failed { .. }) => {
                    let mut suggestion = None;
                    let (span, label, module) = if let PathResult::Failed { span, label, module, .. } = path_res {
                        // try to suggest if it's not a macro, maybe a function
                        if let PathResult::NonModule(partial_res) = self.maybe_resolve_path(&path, Some(ValueNS), &parent_scope)
                            && partial_res.unresolved_segments() == 0 {
                            let sm = self.tcx.sess.source_map();
                            let exclamation_span = sm.next_point(span);
                            suggestion = Some((
                                vec![(exclamation_span, "".to_string())],
                                    format!("{} is not a macro, but a {}, try to remove `!`", Segment::names_to_string(&path), partial_res.base_res().descr()),
                                    Applicability::MaybeIncorrect
                                ));
                        }
                        (span, label, module)
                    } else {
                        (
                            path_span,
                            format!(
                                "partially resolved path in {} {}",
                                kind.article(),
                                kind.descr()
                            ),
                            None,
                        )
                    };
                    self.report_error(
                        span,
                        ResolutionError::FailedToResolve { last_segment: path.last().map(|segment| segment.ident.name), label, suggestion, module },
                    );
                }
                PathResult::Module(..) | PathResult::Indeterminate => unreachable!(),
            }
        }

        let macro_resolutions = mem::take(&mut self.single_segment_macro_resolutions);
        for (ident, kind, parent_scope, initial_binding) in macro_resolutions {
            match self.early_resolve_ident_in_lexical_scope(
                ident,
                ScopeSet::Macro(kind),
                &parent_scope,
                Some(Finalize::new(ast::CRATE_NODE_ID, ident.span)),
                true,
                None,
            ) {
                Ok(binding) => {
                    let initial_res = initial_binding.map(|initial_binding| {
                        self.record_use(ident, initial_binding, false);
                        initial_binding.res()
                    });
                    let res = binding.res();
                    let seg = Segment::from_ident(ident);
                    check_consistency(self, &[seg], ident.span, kind, initial_res, res);
                    if res == Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat) {
                        let node_id = self
                            .invocation_parents
                            .get(&parent_scope.expansion)
                            .map_or(ast::CRATE_NODE_ID, |id| self.def_id_to_node_id[id.0]);
                        self.lint_buffer.buffer_lint_with_diagnostic(
                            LEGACY_DERIVE_HELPERS,
                            node_id,
                            ident.span,
                            "derive helper attribute is used before it is introduced",
                            BuiltinLintDiagnostics::LegacyDeriveHelpers(binding.span),
                        );
                    }
                }
                Err(..) => {
                    let expected = kind.descr_expected();

                    let mut err = self.tcx.sess.create_err(CannotFindIdentInThisScope {
                        span: ident.span,
                        expected,
                        ident,
                    });

                    self.unresolved_macro_suggestions(&mut err, kind, &parent_scope, ident, krate);
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
            if let StabilityLevel::Unstable { reason, issue, is_soft, implied_by } = stability.level
            {
                let feature = stability.feature;

                let is_allowed = |feature| {
                    self.active_features.contains(&feature) || span.allows_unstable(feature)
                };
                let allowed_by_implication = implied_by.is_some_and(|feature| is_allowed(feature));
                if !is_allowed(feature) && !allowed_by_implication {
                    let lint_buffer = &mut self.lint_buffer;
                    let soft_handler =
                        |lint, span, msg: String| lint_buffer.buffer_lint(lint, node_id, span, msg);
                    stability::report_unstable(
                        self.tcx.sess,
                        feature,
                        reason.to_opt_reason(),
                        issue,
                        None,
                        is_soft,
                        span,
                        soft_handler,
                    );
                }
            }
        }
        if let Some(depr) = &ext.deprecation {
            let path = pprust::path_to_string(&path);
            let (message, lint) = stability::deprecation_message_and_lint(depr, "macro", &path);
            stability::early_report_deprecation(
                &mut self.lint_buffer,
                message,
                depr.suggestion,
                lint,
                span,
                node_id,
            );
        }
    }

    fn prohibit_imported_non_macro_attrs(
        &self,
        binding: Option<NameBinding<'a>>,
        res: Option<Res>,
        span: Span,
    ) {
        if let Some(Res::NonMacroAttr(kind)) = res {
            if kind != NonMacroAttrKind::Tool && binding.map_or(true, |b| b.is_import()) {
                let msg =
                    format!("cannot use {} {} through an import", kind.article(), kind.descr());
                let mut err = self.tcx.sess.struct_span_err(span, msg);
                if let Some(binding) = binding {
                    err.span_note(binding.span, format!("the {} imported here", kind.descr()));
                }
                err.emit();
            }
        }
    }

    pub(crate) fn check_reserved_macro_name(&mut self, ident: Ident, res: Res) {
        // Reserve some names that are not quite covered by the general check
        // performed on `Resolver::builtin_attrs`.
        if ident.name == sym::cfg || ident.name == sym::cfg_attr {
            let macro_kind = self.get_macro(res).map(|macro_data| macro_data.ext.macro_kind());
            if macro_kind.is_some() && sub_namespace_match(macro_kind, Some(MacroKind::Attr)) {
                self.tcx.sess.span_err(
                    ident.span,
                    format!("name `{}` is reserved in attribute namespace", ident),
                );
            }
        }
    }

    /// Compile the macro into a `SyntaxExtension` and its rule spans.
    ///
    /// Possibly replace its expander to a pre-defined one for built-in macros.
    pub(crate) fn compile_macro(
        &mut self,
        item: &ast::Item,
        edition: Edition,
    ) -> (SyntaxExtension, Vec<(usize, Span)>) {
        let (mut result, mut rule_spans) = compile_declarative_macro(self.tcx.sess, item, edition);

        if let Some(builtin_name) = result.builtin_name {
            // The macro was marked with `#[rustc_builtin_macro]`.
            if let Some(builtin_macro) = self.builtin_macros.get_mut(&builtin_name) {
                // The macro is a built-in, replace its expander function
                // while still taking everything else from the source code.
                // If we already loaded this builtin macro, give a better error message than 'no such builtin macro'.
                match mem::replace(builtin_macro, BuiltinMacroState::AlreadySeen(item.span)) {
                    BuiltinMacroState::NotYetSeen(ext) => {
                        result.kind = ext;
                        rule_spans = Vec::new();
                        if item.id != ast::DUMMY_NODE_ID {
                            self.builtin_macro_kinds
                                .insert(self.local_def_id(item.id), result.macro_kind());
                        }
                    }
                    BuiltinMacroState::AlreadySeen(span) => {
                        struct_span_err!(
                            self.tcx.sess,
                            item.span,
                            E0773,
                            "attempted to define built-in macro more than once"
                        )
                        .span_note(span, "previously defined here")
                        .emit();
                    }
                }
            } else {
                let msg = format!("cannot find a built-in macro with name `{}`", item.ident);
                self.tcx.sess.span_err(item.span, msg);
            }
        }

        (result, rule_spans)
    }
}
