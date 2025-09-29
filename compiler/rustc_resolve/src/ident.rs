use Determinacy::*;
use Namespace::*;
use rustc_ast::{self as ast, NodeId};
use rustc_errors::ErrorGuaranteed;
use rustc_hir::def::{DefKind, MacroKinds, Namespace, NonMacroAttrKind, PartialRes, PerNS};
use rustc_middle::bug;
use rustc_session::lint::builtin::PROC_MACRO_DERIVE_RESOLUTION_FALLBACK;
use rustc_session::parse::feature_err;
use rustc_span::hygiene::{ExpnId, ExpnKind, LocalExpnId, MacroKind, SyntaxContext};
use rustc_span::{Ident, Span, kw, sym};
use tracing::{debug, instrument};

use crate::errors::{ParamKindInEnumDiscriminant, ParamKindInNonTrivialAnonConst};
use crate::imports::{Import, NameResolution};
use crate::late::{ConstantHasGenerics, NoConstantGenericsReason, PathSource, Rib, RibKind};
use crate::macros::{MacroRulesScope, sub_namespace_match};
use crate::{
    AmbiguityError, AmbiguityErrorMisc, AmbiguityKind, BindingKey, CmResolver, Determinacy,
    Finalize, ImportKind, LexicalScopeBinding, Module, ModuleKind, ModuleOrUniformRoot,
    NameBinding, NameBindingKind, ParentScope, PathResult, PrivacyError, Res, ResolutionError,
    Resolver, Scope, ScopeSet, Segment, Stage, Used, Weak, errors,
};

#[derive(Copy, Clone)]
pub enum UsePrelude {
    No,
    Yes,
}

impl From<UsePrelude> for bool {
    fn from(up: UsePrelude) -> bool {
        matches!(up, UsePrelude::Yes)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Shadowing {
    Restricted,
    Unrestricted,
}

impl<'ra, 'tcx> Resolver<'ra, 'tcx> {
    /// A generic scope visitor.
    /// Visits scopes in order to resolve some identifier in them or perform other actions.
    /// If the callback returns `Some` result, we stop visiting scopes and return it.
    pub(crate) fn visit_scopes<'r, T>(
        mut self: CmResolver<'r, 'ra, 'tcx>,
        scope_set: ScopeSet<'ra>,
        parent_scope: &ParentScope<'ra>,
        ctxt: SyntaxContext,
        derive_fallback_lint_id: Option<NodeId>,
        mut visitor: impl FnMut(
            &mut CmResolver<'r, 'ra, 'tcx>,
            Scope<'ra>,
            UsePrelude,
            SyntaxContext,
        ) -> Option<T>,
    ) -> Option<T> {
        // General principles:
        // 1. Not controlled (user-defined) names should have higher priority than controlled names
        //    built into the language or standard library. This way we can add new names into the
        //    language or standard library without breaking user code.
        // 2. "Closed set" below means new names cannot appear after the current resolution attempt.
        // Places to search (in order of decreasing priority):
        // (Type NS)
        // 1. FIXME: Ribs (type parameters), there's no necessary infrastructure yet
        //    (open set, not controlled).
        // 2. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled).
        // 3. Extern prelude (open, the open part is from macro expansions, not controlled).
        // 4. Tool modules (closed, controlled right now, but not in the future).
        // 5. Standard library prelude (de-facto closed, controlled).
        // 6. Language prelude (closed, controlled).
        // (Value NS)
        // 1. FIXME: Ribs (local variables), there's no necessary infrastructure yet
        //    (open set, not controlled).
        // 2. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled).
        // 3. Standard library prelude (de-facto closed, controlled).
        // (Macro NS)
        // 1-3. Derive helpers (open, not controlled). All ambiguities with other names
        //    are currently reported as errors. They should be higher in priority than preludes
        //    and probably even names in modules according to the "general principles" above. They
        //    also should be subject to restricted shadowing because are effectively produced by
        //    derives (you need to resolve the derive first to add helpers into scope), but they
        //    should be available before the derive is expanded for compatibility.
        //    It's mess in general, so we are being conservative for now.
        // 1-3. `macro_rules` (open, not controlled), loop through `macro_rules` scopes. Have higher
        //    priority than prelude macros, but create ambiguities with macros in modules.
        // 1-3. Names in modules (both normal `mod`ules and blocks), loop through hygienic parents
        //    (open, not controlled). Have higher priority than prelude macros, but create
        //    ambiguities with `macro_rules`.
        // 4. `macro_use` prelude (open, the open part is from macro expansions, not controlled).
        // 4a. User-defined prelude from macro-use
        //    (open, the open part is from macro expansions, not controlled).
        // 4b. "Standard library prelude" part implemented through `macro-use` (closed, controlled).
        // 4c. Standard library prelude (de-facto closed, controlled).
        // 6. Language prelude: builtin attributes (closed, controlled).

        let rust_2015 = ctxt.edition().is_rust_2015();
        let (ns, macro_kind) = match scope_set {
            ScopeSet::All(ns) | ScopeSet::ModuleAndExternPrelude(ns, _) => (ns, None),
            ScopeSet::ExternPrelude => (TypeNS, None),
            ScopeSet::Macro(macro_kind) => (MacroNS, Some(macro_kind)),
        };
        let module = match scope_set {
            // Start with the specified module.
            ScopeSet::ModuleAndExternPrelude(_, module) => module,
            // Jump out of trait or enum modules, they do not act as scopes.
            _ => parent_scope.module.nearest_item_scope(),
        };
        let module_and_extern_prelude = matches!(scope_set, ScopeSet::ModuleAndExternPrelude(..));
        let extern_prelude = matches!(scope_set, ScopeSet::ExternPrelude);
        let mut scope = match ns {
            _ if module_and_extern_prelude => Scope::Module(module, None),
            _ if extern_prelude => Scope::ExternPreludeItems,
            TypeNS | ValueNS => Scope::Module(module, None),
            MacroNS => Scope::DeriveHelpers(parent_scope.expansion),
        };
        let mut ctxt = ctxt.normalize_to_macros_2_0();
        let mut use_prelude = !module.no_implicit_prelude;

        loop {
            let visit = match scope {
                // Derive helpers are not in scope when resolving derives in the same container.
                Scope::DeriveHelpers(expn_id) => {
                    !(expn_id == parent_scope.expansion && macro_kind == Some(MacroKind::Derive))
                }
                Scope::DeriveHelpersCompat => true,
                Scope::MacroRules(macro_rules_scope) => {
                    // Use "path compression" on `macro_rules` scope chains. This is an optimization
                    // used to avoid long scope chains, see the comments on `MacroRulesScopeRef`.
                    // As another consequence of this optimization visitors never observe invocation
                    // scopes for macros that were already expanded.
                    while let MacroRulesScope::Invocation(invoc_id) = macro_rules_scope.get() {
                        if let Some(next_scope) = self.output_macro_rules_scopes.get(&invoc_id) {
                            macro_rules_scope.set(next_scope.get());
                        } else {
                            break;
                        }
                    }
                    true
                }
                Scope::Module(..) => true,
                Scope::MacroUsePrelude => use_prelude || rust_2015,
                Scope::BuiltinAttrs => true,
                Scope::ExternPreludeItems | Scope::ExternPreludeFlags => {
                    use_prelude || module_and_extern_prelude || extern_prelude
                }
                Scope::ToolPrelude => use_prelude,
                Scope::StdLibPrelude => use_prelude || ns == MacroNS,
                Scope::BuiltinTypes => true,
            };

            if visit {
                let use_prelude = if use_prelude { UsePrelude::Yes } else { UsePrelude::No };
                if let break_result @ Some(..) = visitor(&mut self, scope, use_prelude, ctxt) {
                    return break_result;
                }
            }

            scope = match scope {
                Scope::DeriveHelpers(LocalExpnId::ROOT) => Scope::DeriveHelpersCompat,
                Scope::DeriveHelpers(expn_id) => {
                    // Derive helpers are not visible to code generated by bang or derive macros.
                    let expn_data = expn_id.expn_data();
                    match expn_data.kind {
                        ExpnKind::Root
                        | ExpnKind::Macro(MacroKind::Bang | MacroKind::Derive, _) => {
                            Scope::DeriveHelpersCompat
                        }
                        _ => Scope::DeriveHelpers(expn_data.parent.expect_local()),
                    }
                }
                Scope::DeriveHelpersCompat => Scope::MacroRules(parent_scope.macro_rules),
                Scope::MacroRules(macro_rules_scope) => match macro_rules_scope.get() {
                    MacroRulesScope::Binding(binding) => {
                        Scope::MacroRules(binding.parent_macro_rules_scope)
                    }
                    MacroRulesScope::Invocation(invoc_id) => {
                        Scope::MacroRules(self.invocation_parent_scopes[&invoc_id].macro_rules)
                    }
                    MacroRulesScope::Empty => Scope::Module(module, None),
                },
                Scope::Module(..) if module_and_extern_prelude => match ns {
                    TypeNS => {
                        ctxt.adjust(ExpnId::root());
                        Scope::ExternPreludeItems
                    }
                    ValueNS | MacroNS => break,
                },
                Scope::Module(module, prev_lint_id) => {
                    use_prelude = !module.no_implicit_prelude;
                    match self.hygienic_lexical_parent(module, &mut ctxt, derive_fallback_lint_id) {
                        Some((parent_module, lint_id)) => {
                            Scope::Module(parent_module, lint_id.or(prev_lint_id))
                        }
                        None => {
                            ctxt.adjust(ExpnId::root());
                            match ns {
                                TypeNS => Scope::ExternPreludeItems,
                                ValueNS => Scope::StdLibPrelude,
                                MacroNS => Scope::MacroUsePrelude,
                            }
                        }
                    }
                }
                Scope::MacroUsePrelude => Scope::StdLibPrelude,
                Scope::BuiltinAttrs => break, // nowhere else to search
                Scope::ExternPreludeItems => Scope::ExternPreludeFlags,
                Scope::ExternPreludeFlags if module_and_extern_prelude || extern_prelude => break,
                Scope::ExternPreludeFlags => Scope::ToolPrelude,
                Scope::ToolPrelude => Scope::StdLibPrelude,
                Scope::StdLibPrelude => match ns {
                    TypeNS => Scope::BuiltinTypes,
                    ValueNS => break, // nowhere else to search
                    MacroNS => Scope::BuiltinAttrs,
                },
                Scope::BuiltinTypes => break, // nowhere else to search
            };
        }

        None
    }

    fn hygienic_lexical_parent(
        &self,
        module: Module<'ra>,
        ctxt: &mut SyntaxContext,
        derive_fallback_lint_id: Option<NodeId>,
    ) -> Option<(Module<'ra>, Option<NodeId>)> {
        if !module.expansion.outer_expn_is_descendant_of(*ctxt) {
            return Some((self.expn_def_scope(ctxt.remove_mark()), None));
        }

        if let ModuleKind::Block = module.kind {
            return Some((module.parent.unwrap().nearest_item_scope(), None));
        }

        // We need to support the next case under a deprecation warning
        // ```
        // struct MyStruct;
        // ---- begin: this comes from a proc macro derive
        // mod implementation_details {
        //     // Note that `MyStruct` is not in scope here.
        //     impl SomeTrait for MyStruct { ... }
        // }
        // ---- end
        // ```
        // So we have to fall back to the module's parent during lexical resolution in this case.
        if derive_fallback_lint_id.is_some()
            && let Some(parent) = module.parent
            // Inner module is inside the macro
            && module.expansion != parent.expansion
            // Parent module is outside of the macro
            && module.expansion.is_descendant_of(parent.expansion)
            // The macro is a proc macro derive
            && let Some(def_id) = module.expansion.expn_data().macro_def_id
        {
            let ext = &self.get_macro_by_def_id(def_id).ext;
            if ext.builtin_name.is_none()
                && ext.macro_kinds() == MacroKinds::DERIVE
                && parent.expansion.outer_expn_is_descendant_of(*ctxt)
            {
                return Some((parent, derive_fallback_lint_id));
            }
        }

        None
    }

    /// This resolves the identifier `ident` in the namespace `ns` in the current lexical scope.
    /// More specifically, we proceed up the hierarchy of scopes and return the binding for
    /// `ident` in the first scope that defines it (or None if no scopes define it).
    ///
    /// A block's items are above its local variables in the scope hierarchy, regardless of where
    /// the items are defined in the block. For example,
    /// ```rust
    /// fn f() {
    ///    g(); // Since there are no local variables in scope yet, this resolves to the item.
    ///    let g = || {};
    ///    fn g() {}
    ///    g(); // This resolves to the local variable `g` since it shadows the item.
    /// }
    /// ```
    ///
    /// Invariant: This must only be called during main resolution, not during
    /// import resolution.
    #[instrument(level = "debug", skip(self, ribs))]
    pub(crate) fn resolve_ident_in_lexical_scope(
        &mut self,
        mut ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'ra>,
        finalize: Option<Finalize>,
        ribs: &[Rib<'ra>],
        ignore_binding: Option<NameBinding<'ra>>,
    ) -> Option<LexicalScopeBinding<'ra>> {
        assert!(ns == TypeNS || ns == ValueNS);
        let orig_ident = ident;
        let (general_span, normalized_span) = if ident.name == kw::SelfUpper {
            // FIXME(jseyfried) improve `Self` hygiene
            let empty_span = ident.span.with_ctxt(SyntaxContext::root());
            (empty_span, empty_span)
        } else if ns == TypeNS {
            let normalized_span = ident.span.normalize_to_macros_2_0();
            (normalized_span, normalized_span)
        } else {
            (ident.span.normalize_to_macro_rules(), ident.span.normalize_to_macros_2_0())
        };
        ident.span = general_span;
        let normalized_ident = Ident { span: normalized_span, ..ident };

        // Walk backwards up the ribs in scope.
        for (i, rib) in ribs.iter().enumerate().rev() {
            debug!("walk rib\n{:?}", rib.bindings);
            // Use the rib kind to determine whether we are resolving parameters
            // (macro 2.0 hygiene) or local variables (`macro_rules` hygiene).
            let rib_ident = if rib.kind.contains_params() { normalized_ident } else { ident };
            if let Some((original_rib_ident_def, res)) = rib.bindings.get_key_value(&rib_ident) {
                // The ident resolves to a type parameter or local variable.
                return Some(LexicalScopeBinding::Res(self.validate_res_from_ribs(
                    i,
                    rib_ident,
                    *res,
                    finalize.map(|finalize| finalize.path_span),
                    *original_rib_ident_def,
                    ribs,
                )));
            } else if let RibKind::Block(Some(module)) = rib.kind
                && let Ok(binding) = self.cm().resolve_ident_in_module_unadjusted(
                    ModuleOrUniformRoot::Module(module),
                    ident,
                    ns,
                    parent_scope,
                    Shadowing::Unrestricted,
                    finalize.map(|finalize| Finalize { used: Used::Scope, ..finalize }),
                    ignore_binding,
                    None,
                )
            {
                // The ident resolves to an item in a block.
                return Some(LexicalScopeBinding::Item(binding));
            } else if let RibKind::Module(module) = rib.kind {
                // Encountered a module item, abandon ribs and look into that module and preludes.
                let parent_scope = &ParentScope { module, ..*parent_scope };
                let finalize = finalize.map(|f| Finalize { stage: Stage::Late, ..f });
                return self
                    .cm()
                    .resolve_ident_in_scope_set(
                        orig_ident,
                        ScopeSet::All(ns),
                        parent_scope,
                        finalize,
                        finalize.is_some(),
                        ignore_binding,
                        None,
                    )
                    .ok()
                    .map(LexicalScopeBinding::Item);
            }

            if let RibKind::MacroDefinition(def) = rib.kind
                && def == self.macro_def(ident.span.ctxt())
            {
                // If an invocation of this macro created `ident`, give up on `ident`
                // and switch to `ident`'s source from the macro definition.
                ident.span.remove_mark();
            }
        }

        unreachable!()
    }

    /// Resolve an identifier in the specified set of scopes.
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn resolve_ident_in_scope_set<'r>(
        self: CmResolver<'r, 'ra, 'tcx>,
        orig_ident: Ident,
        scope_set: ScopeSet<'ra>,
        parent_scope: &ParentScope<'ra>,
        finalize: Option<Finalize>,
        force: bool,
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
    ) -> Result<NameBinding<'ra>, Determinacy> {
        bitflags::bitflags! {
            #[derive(Clone, Copy)]
            struct Flags: u8 {
                const MACRO_RULES          = 1 << 0;
                const MODULE               = 1 << 1;
                const MISC_SUGGEST_CRATE   = 1 << 2;
                const MISC_SUGGEST_SELF    = 1 << 3;
                const MISC_FROM_PRELUDE    = 1 << 4;
            }
        }

        assert!(force || finalize.is_none()); // `finalize` implies `force`

        // Make sure `self`, `super` etc produce an error when passed to here.
        if orig_ident.is_path_segment_keyword() {
            return Err(Determinacy::Determined);
        }

        let (ns, macro_kind) = match scope_set {
            ScopeSet::All(ns) | ScopeSet::ModuleAndExternPrelude(ns, _) => (ns, None),
            ScopeSet::ExternPrelude => (TypeNS, None),
            ScopeSet::Macro(macro_kind) => (MacroNS, Some(macro_kind)),
        };

        // This is *the* result, resolution from the scope closest to the resolved identifier.
        // However, sometimes this result is "weak" because it comes from a glob import or
        // a macro expansion, and in this case it cannot shadow names from outer scopes, e.g.
        // mod m { ... } // solution in outer scope
        // {
        //     use prefix::*; // imports another `m` - innermost solution
        //                    // weak, cannot shadow the outer `m`, need to report ambiguity error
        //     m::mac!();
        // }
        // So we have to save the innermost solution and continue searching in outer scopes
        // to detect potential ambiguities.
        let mut innermost_result: Option<(NameBinding<'_>, Flags)> = None;
        let mut determinacy = Determinacy::Determined;
        let mut extern_prelude_item_binding = None;
        let mut extern_prelude_flag_binding = None;
        // Shadowed bindings don't need to be marked as used or non-speculatively loaded.
        macro finalize_scope() {
            if innermost_result.is_none() { finalize } else { None }
        }

        // Go through all the scopes and try to resolve the name.
        let derive_fallback_lint_id = match finalize {
            Some(Finalize { node_id, stage: Stage::Late, .. }) => Some(node_id),
            _ => None,
        };
        let break_result = self.visit_scopes(
            scope_set,
            parent_scope,
            orig_ident.span.ctxt(),
            derive_fallback_lint_id,
            |this, scope, use_prelude, ctxt| {
                let ident = Ident::new(orig_ident.name, orig_ident.span.with_ctxt(ctxt));
                let result = match scope {
                    Scope::DeriveHelpers(expn_id) => {
                        if let Some(binding) = this.helper_attrs.get(&expn_id).and_then(|attrs| {
                            attrs.iter().rfind(|(i, _)| ident == *i).map(|(_, binding)| *binding)
                        }) {
                            Ok((binding, Flags::empty()))
                        } else {
                            Err(Determinacy::Determined)
                        }
                    }
                    Scope::DeriveHelpersCompat => {
                        let mut result = Err(Determinacy::Determined);
                        for derive in parent_scope.derives {
                            let parent_scope = &ParentScope { derives: &[], ..*parent_scope };
                            match this.reborrow().resolve_macro_path(
                                derive,
                                MacroKind::Derive,
                                parent_scope,
                                true,
                                force,
                                ignore_import,
                                None,
                            ) {
                                Ok((Some(ext), _)) => {
                                    if ext.helper_attrs.contains(&ident.name) {
                                        let binding = this.arenas.new_pub_res_binding(
                                            Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat),
                                            derive.span,
                                            LocalExpnId::ROOT,
                                        );
                                        result = Ok((binding, Flags::empty()));
                                        break;
                                    }
                                }
                                Ok(_) | Err(Determinacy::Determined) => {}
                                Err(Determinacy::Undetermined) => {
                                    result = Err(Determinacy::Undetermined)
                                }
                            }
                        }
                        result
                    }
                    Scope::MacroRules(macro_rules_scope) => match macro_rules_scope.get() {
                        MacroRulesScope::Binding(macro_rules_binding)
                            if ident == macro_rules_binding.ident =>
                        {
                            Ok((macro_rules_binding.binding, Flags::MACRO_RULES))
                        }
                        MacroRulesScope::Invocation(_) => Err(Determinacy::Undetermined),
                        _ => Err(Determinacy::Determined),
                    },
                    Scope::Module(module, derive_fallback_lint_id) => {
                        let (adjusted_parent_scope, adjusted_finalize) =
                            if matches!(scope_set, ScopeSet::ModuleAndExternPrelude(..)) {
                                (parent_scope, finalize_scope!())
                            } else {
                                (
                                    &ParentScope { module, ..*parent_scope },
                                    finalize_scope!().map(|f| Finalize { used: Used::Scope, ..f }),
                                )
                            };
                        let binding = this.reborrow().resolve_ident_in_module_unadjusted(
                            ModuleOrUniformRoot::Module(module),
                            ident,
                            ns,
                            adjusted_parent_scope,
                            Shadowing::Restricted,
                            adjusted_finalize,
                            ignore_binding,
                            ignore_import,
                        );
                        match binding {
                            Ok(binding) => {
                                if let Some(lint_id) = derive_fallback_lint_id {
                                    this.get_mut().lint_buffer.buffer_lint(
                                        PROC_MACRO_DERIVE_RESOLUTION_FALLBACK,
                                        lint_id,
                                        orig_ident.span,
                                        errors::ProcMacroDeriveResolutionFallback {
                                            span: orig_ident.span,
                                            ns_descr: ns.descr(),
                                            ident,
                                        },
                                    );
                                }
                                let misc_flags = if module == this.graph_root {
                                    Flags::MISC_SUGGEST_CRATE
                                } else if module.is_normal() {
                                    Flags::MISC_SUGGEST_SELF
                                } else {
                                    Flags::empty()
                                };
                                Ok((binding, Flags::MODULE | misc_flags))
                            }
                            Err((Determinacy::Undetermined, Weak::No)) => {
                                return Some(Err(Determinacy::determined(force)));
                            }
                            Err((Determinacy::Undetermined, Weak::Yes)) => {
                                Err(Determinacy::Undetermined)
                            }
                            Err((Determinacy::Determined, _)) => Err(Determinacy::Determined),
                        }
                    }
                    Scope::MacroUsePrelude => {
                        match this.macro_use_prelude.get(&ident.name).cloned() {
                            Some(binding) => Ok((binding, Flags::MISC_FROM_PRELUDE)),
                            None => Err(Determinacy::determined(
                                this.graph_root.unexpanded_invocations.borrow().is_empty(),
                            )),
                        }
                    }
                    Scope::BuiltinAttrs => match this.builtin_attrs_bindings.get(&ident.name) {
                        Some(binding) => Ok((*binding, Flags::empty())),
                        None => Err(Determinacy::Determined),
                    },
                    Scope::ExternPreludeItems => {
                        match this
                            .reborrow()
                            .extern_prelude_get_item(ident, finalize_scope!().is_some())
                        {
                            Some(binding) => {
                                extern_prelude_item_binding = Some(binding);
                                Ok((binding, Flags::empty()))
                            }
                            None => Err(Determinacy::determined(
                                this.graph_root.unexpanded_invocations.borrow().is_empty(),
                            )),
                        }
                    }
                    Scope::ExternPreludeFlags => {
                        match this.extern_prelude_get_flag(ident, finalize_scope!().is_some()) {
                            Some(binding) => {
                                extern_prelude_flag_binding = Some(binding);
                                Ok((binding, Flags::empty()))
                            }
                            None => Err(Determinacy::Determined),
                        }
                    }
                    Scope::ToolPrelude => match this.registered_tool_bindings.get(&ident) {
                        Some(binding) => Ok((*binding, Flags::empty())),
                        None => Err(Determinacy::Determined),
                    },
                    Scope::StdLibPrelude => {
                        let mut result = Err(Determinacy::Determined);
                        if let Some(prelude) = this.prelude
                            && let Ok(binding) = this.reborrow().resolve_ident_in_module_unadjusted(
                                ModuleOrUniformRoot::Module(prelude),
                                ident,
                                ns,
                                parent_scope,
                                Shadowing::Unrestricted,
                                None,
                                ignore_binding,
                                ignore_import,
                            )
                            && (matches!(use_prelude, UsePrelude::Yes)
                                || this.is_builtin_macro(binding.res()))
                        {
                            result = Ok((binding, Flags::MISC_FROM_PRELUDE));
                        }

                        result
                    }
                    Scope::BuiltinTypes => match this.builtin_types_bindings.get(&ident.name) {
                        Some(binding) => {
                            if matches!(ident.name, sym::f16)
                                && !this.tcx.features().f16()
                                && !ident.span.allows_unstable(sym::f16)
                                && finalize_scope!().is_some()
                            {
                                feature_err(
                                    this.tcx.sess,
                                    sym::f16,
                                    ident.span,
                                    "the type `f16` is unstable",
                                )
                                .emit();
                            }
                            if matches!(ident.name, sym::f128)
                                && !this.tcx.features().f128()
                                && !ident.span.allows_unstable(sym::f128)
                                && finalize_scope!().is_some()
                            {
                                feature_err(
                                    this.tcx.sess,
                                    sym::f128,
                                    ident.span,
                                    "the type `f128` is unstable",
                                )
                                .emit();
                            }
                            Ok((*binding, Flags::empty()))
                        }
                        None => Err(Determinacy::Determined),
                    },
                };

                match result {
                    Ok((binding, flags)) => {
                        if !sub_namespace_match(binding.macro_kinds(), macro_kind) {
                            return None;
                        }

                        // Below we report various ambiguity errors.
                        // We do not need to report them if we are either in speculative resolution,
                        // or in late resolution when everything is already imported and expanded
                        // and no ambiguities exist.
                        if matches!(finalize, None | Some(Finalize { stage: Stage::Late, .. })) {
                            return Some(Ok(binding));
                        }

                        if let Some((innermost_binding, innermost_flags)) = innermost_result {
                            // Found another solution, if the first one was "weak", report an error.
                            let (res, innermost_res) = (binding.res(), innermost_binding.res());
                            if res != innermost_res {
                                let is_builtin = |res| {
                                    matches!(res, Res::NonMacroAttr(NonMacroAttrKind::Builtin(..)))
                                };
                                let derive_helper =
                                    Res::NonMacroAttr(NonMacroAttrKind::DeriveHelper);
                                let derive_helper_compat =
                                    Res::NonMacroAttr(NonMacroAttrKind::DeriveHelperCompat);

                                let ambiguity_error_kind = if is_builtin(innermost_res)
                                    || is_builtin(res)
                                {
                                    Some(AmbiguityKind::BuiltinAttr)
                                } else if innermost_res == derive_helper_compat
                                    || res == derive_helper_compat && innermost_res != derive_helper
                                {
                                    Some(AmbiguityKind::DeriveHelper)
                                } else if innermost_flags.contains(Flags::MACRO_RULES)
                                    && flags.contains(Flags::MODULE)
                                    && !this.disambiguate_macro_rules_vs_modularized(
                                        innermost_binding,
                                        binding,
                                    )
                                    || flags.contains(Flags::MACRO_RULES)
                                        && innermost_flags.contains(Flags::MODULE)
                                        && !this.disambiguate_macro_rules_vs_modularized(
                                            binding,
                                            innermost_binding,
                                        )
                                {
                                    Some(AmbiguityKind::MacroRulesVsModularized)
                                } else if innermost_binding.is_glob_import() {
                                    Some(AmbiguityKind::GlobVsOuter)
                                } else if innermost_binding
                                    .may_appear_after(parent_scope.expansion, binding)
                                {
                                    Some(AmbiguityKind::MoreExpandedVsOuter)
                                } else {
                                    None
                                };
                                // Skip ambiguity errors for extern flag bindings "overridden"
                                // by extern item bindings.
                                // FIXME: Remove with lang team approval.
                                let issue_145575_hack = Some(binding)
                                    == extern_prelude_flag_binding
                                    && extern_prelude_item_binding.is_some()
                                    && extern_prelude_item_binding != Some(innermost_binding);
                                if let Some(kind) = ambiguity_error_kind
                                    && !issue_145575_hack
                                {
                                    let misc = |f: Flags| {
                                        if f.contains(Flags::MISC_SUGGEST_CRATE) {
                                            AmbiguityErrorMisc::SuggestCrate
                                        } else if f.contains(Flags::MISC_SUGGEST_SELF) {
                                            AmbiguityErrorMisc::SuggestSelf
                                        } else if f.contains(Flags::MISC_FROM_PRELUDE) {
                                            AmbiguityErrorMisc::FromPrelude
                                        } else {
                                            AmbiguityErrorMisc::None
                                        }
                                    };
                                    this.get_mut().ambiguity_errors.push(AmbiguityError {
                                        kind,
                                        ident: orig_ident,
                                        b1: innermost_binding,
                                        b2: binding,
                                        warning: false,
                                        misc1: misc(innermost_flags),
                                        misc2: misc(flags),
                                    });
                                    return Some(Ok(innermost_binding));
                                }
                            }
                        } else {
                            // Found the first solution.
                            innermost_result = Some((binding, flags));
                        }
                    }
                    Err(Determinacy::Determined) => {}
                    Err(Determinacy::Undetermined) => determinacy = Determinacy::Undetermined,
                }

                None
            },
        );

        if let Some(break_result) = break_result {
            return break_result;
        }

        // The first found solution was the only one, return it.
        if let Some((binding, _)) = innermost_result {
            return Ok(binding);
        }

        Err(Determinacy::determined(determinacy == Determinacy::Determined || force))
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn maybe_resolve_ident_in_module<'r>(
        self: CmResolver<'r, 'ra, 'tcx>,
        module: ModuleOrUniformRoot<'ra>,
        ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'ra>,
        ignore_import: Option<Import<'ra>>,
    ) -> Result<NameBinding<'ra>, Determinacy> {
        self.resolve_ident_in_module(module, ident, ns, parent_scope, None, None, ignore_import)
            .map_err(|(determinacy, _)| determinacy)
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn resolve_ident_in_module<'r>(
        self: CmResolver<'r, 'ra, 'tcx>,
        module: ModuleOrUniformRoot<'ra>,
        mut ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'ra>,
        finalize: Option<Finalize>,
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
    ) -> Result<NameBinding<'ra>, (Determinacy, Weak)> {
        let tmp_parent_scope;
        let mut adjusted_parent_scope = parent_scope;
        match module {
            ModuleOrUniformRoot::Module(m) => {
                if let Some(def) = ident.span.normalize_to_macros_2_0_and_adjust(m.expansion) {
                    tmp_parent_scope =
                        ParentScope { module: self.expn_def_scope(def), ..*parent_scope };
                    adjusted_parent_scope = &tmp_parent_scope;
                }
            }
            ModuleOrUniformRoot::ExternPrelude => {
                ident.span.normalize_to_macros_2_0_and_adjust(ExpnId::root());
            }
            ModuleOrUniformRoot::ModuleAndExternPrelude(..) | ModuleOrUniformRoot::CurrentScope => {
                // No adjustments
            }
        }
        self.resolve_ident_in_module_unadjusted(
            module,
            ident,
            ns,
            adjusted_parent_scope,
            Shadowing::Unrestricted,
            finalize,
            ignore_binding,
            ignore_import,
        )
    }
    /// Attempts to resolve `ident` in namespaces `ns` of `module`.
    /// Invariant: if `finalize` is `Some`, expansion and import resolution must be complete.
    #[instrument(level = "debug", skip(self))]
    fn resolve_ident_in_module_unadjusted<'r>(
        mut self: CmResolver<'r, 'ra, 'tcx>,
        module: ModuleOrUniformRoot<'ra>,
        ident: Ident,
        ns: Namespace,
        parent_scope: &ParentScope<'ra>,
        shadowing: Shadowing,
        finalize: Option<Finalize>,
        // This binding should be ignored during in-module resolution, so that we don't get
        // "self-confirming" import resolutions during import validation and checking.
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
    ) -> Result<NameBinding<'ra>, (Determinacy, Weak)> {
        let module = match module {
            ModuleOrUniformRoot::Module(module) => module,
            ModuleOrUniformRoot::ModuleAndExternPrelude(module) => {
                assert_eq!(shadowing, Shadowing::Unrestricted);
                let binding = self.resolve_ident_in_scope_set(
                    ident,
                    ScopeSet::ModuleAndExternPrelude(ns, module),
                    parent_scope,
                    finalize,
                    finalize.is_some(),
                    ignore_binding,
                    ignore_import,
                );
                return binding.map_err(|determinacy| (determinacy, Weak::No));
            }
            ModuleOrUniformRoot::ExternPrelude => {
                assert_eq!(shadowing, Shadowing::Unrestricted);
                return if ns != TypeNS {
                    Err((Determined, Weak::No))
                } else {
                    let binding = self.resolve_ident_in_scope_set(
                        ident,
                        ScopeSet::ExternPrelude,
                        parent_scope,
                        finalize,
                        finalize.is_some(),
                        ignore_binding,
                        ignore_import,
                    );
                    return binding.map_err(|determinacy| (determinacy, Weak::No));
                };
            }
            ModuleOrUniformRoot::CurrentScope => {
                assert_eq!(shadowing, Shadowing::Unrestricted);
                if ns == TypeNS {
                    if ident.name == kw::Crate || ident.name == kw::DollarCrate {
                        let module = self.resolve_crate_root(ident);
                        return Ok(module.self_binding.unwrap());
                    } else if ident.name == kw::Super || ident.name == kw::SelfLower {
                        // FIXME: Implement these with renaming requirements so that e.g.
                        // `use super;` doesn't work, but `use super as name;` does.
                        // Fall through here to get an error from `early_resolve_...`.
                    }
                }

                let binding = self.resolve_ident_in_scope_set(
                    ident,
                    ScopeSet::All(ns),
                    parent_scope,
                    finalize,
                    finalize.is_some(),
                    ignore_binding,
                    ignore_import,
                );
                return binding.map_err(|determinacy| (determinacy, Weak::No));
            }
        };

        let key = BindingKey::new(ident, ns);
        // `try_borrow_mut` is required to ensure exclusive access, even if the resulting binding
        // doesn't need to be mutable. It will fail when there is a cycle of imports, and without
        // the exclusive access infinite recursion will crash the compiler with stack overflow.
        let resolution = &*self
            .resolution_or_default(module, key)
            .try_borrow_mut()
            .map_err(|_| (Determined, Weak::No))?;

        // If the primary binding is unusable, search further and return the shadowed glob
        // binding if it exists. What we really want here is having two separate scopes in
        // a module - one for non-globs and one for globs, but until that's done use this
        // hack to avoid inconsistent resolution ICEs during import validation.
        let binding = [resolution.non_glob_binding, resolution.glob_binding]
            .into_iter()
            .find_map(|binding| if binding == ignore_binding { None } else { binding });

        if let Some(finalize) = finalize {
            return self.get_mut().finalize_module_binding(
                ident,
                binding,
                if resolution.non_glob_binding.is_some() { resolution.glob_binding } else { None },
                parent_scope,
                module,
                finalize,
                shadowing,
            );
        }

        let check_usable = |this: CmResolver<'r, 'ra, 'tcx>, binding: NameBinding<'ra>| {
            let usable = this.is_accessible_from(binding.vis, parent_scope.module);
            if usable { Ok(binding) } else { Err((Determined, Weak::No)) }
        };

        // Items and single imports are not shadowable, if we have one, then it's determined.
        if let Some(binding) = binding
            && !binding.is_glob_import()
        {
            return check_usable(self, binding);
        }

        // --- From now on we either have a glob resolution or no resolution. ---

        // Check if one of single imports can still define the name,
        // if it can then our result is not determined and can be invalidated.
        if self.reborrow().single_import_can_define_name(
            &resolution,
            binding,
            ns,
            ignore_import,
            ignore_binding,
            parent_scope,
        ) {
            return Err((Undetermined, Weak::No));
        }

        // So we have a resolution that's from a glob import. This resolution is determined
        // if it cannot be shadowed by some new item/import expanded from a macro.
        // This happens either if there are no unexpanded macros, or expanded names cannot
        // shadow globs (that happens in macro namespace or with restricted shadowing).
        //
        // Additionally, any macro in any module can plant names in the root module if it creates
        // `macro_export` macros, so the root module effectively has unresolved invocations if any
        // module has unresolved invocations.
        // However, it causes resolution/expansion to stuck too often (#53144), so, to make
        // progress, we have to ignore those potential unresolved invocations from other modules
        // and prohibit access to macro-expanded `macro_export` macros instead (unless restricted
        // shadowing is enabled, see `macro_expanded_macro_export_errors`).
        if let Some(binding) = binding {
            if binding.determined() || ns == MacroNS || shadowing == Shadowing::Restricted {
                return check_usable(self, binding);
            } else {
                return Err((Undetermined, Weak::No));
            }
        }

        // --- From now on we have no resolution. ---

        // Now we are in situation when new item/import can appear only from a glob or a macro
        // expansion. With restricted shadowing names from globs and macro expansions cannot
        // shadow names from outer scopes, so we can freely fallback from module search to search
        // in outer scopes. For `resolve_ident_in_scope_set` to continue search in outer
        // scopes we return `Undetermined` with `Weak::Yes`.

        // Check if one of unexpanded macros can still define the name,
        // if it can then our "no resolution" result is not determined and can be invalidated.
        if !module.unexpanded_invocations.borrow().is_empty() {
            return Err((Undetermined, Weak::Yes));
        }

        // Check if one of glob imports can still define the name,
        // if it can then our "no resolution" result is not determined and can be invalidated.
        for glob_import in module.globs.borrow().iter() {
            if ignore_import == Some(*glob_import) {
                continue;
            }
            if !self.is_accessible_from(glob_import.vis, parent_scope.module) {
                continue;
            }
            let module = match glob_import.imported_module.get() {
                Some(ModuleOrUniformRoot::Module(module)) => module,
                Some(_) => continue,
                None => return Err((Undetermined, Weak::Yes)),
            };
            let tmp_parent_scope;
            let (mut adjusted_parent_scope, mut ident) =
                (parent_scope, ident.normalize_to_macros_2_0());
            match ident.span.glob_adjust(module.expansion, glob_import.span) {
                Some(Some(def)) => {
                    tmp_parent_scope =
                        ParentScope { module: self.expn_def_scope(def), ..*parent_scope };
                    adjusted_parent_scope = &tmp_parent_scope;
                }
                Some(None) => {}
                None => continue,
            };
            let result = self.reborrow().resolve_ident_in_module_unadjusted(
                ModuleOrUniformRoot::Module(module),
                ident,
                ns,
                adjusted_parent_scope,
                Shadowing::Unrestricted,
                None,
                ignore_binding,
                ignore_import,
            );

            match result {
                Err((Determined, _)) => continue,
                Ok(binding)
                    if !self.is_accessible_from(binding.vis, glob_import.parent_scope.module) =>
                {
                    continue;
                }
                Ok(_) | Err((Undetermined, _)) => return Err((Undetermined, Weak::Yes)),
            }
        }

        // No resolution and no one else can define the name - determinate error.
        Err((Determined, Weak::No))
    }

    fn finalize_module_binding(
        &mut self,
        ident: Ident,
        binding: Option<NameBinding<'ra>>,
        shadowed_glob: Option<NameBinding<'ra>>,
        parent_scope: &ParentScope<'ra>,
        module: Module<'ra>,
        finalize: Finalize,
        shadowing: Shadowing,
    ) -> Result<NameBinding<'ra>, (Determinacy, Weak)> {
        let Finalize { path_span, report_private, used, root_span, .. } = finalize;

        let Some(binding) = binding else {
            return Err((Determined, Weak::No));
        };

        if !self.is_accessible_from(binding.vis, parent_scope.module) {
            if report_private {
                self.privacy_errors.push(PrivacyError {
                    ident,
                    binding,
                    dedup_span: path_span,
                    outermost_res: None,
                    source: None,
                    parent_scope: *parent_scope,
                    single_nested: path_span != root_span,
                });
            } else {
                return Err((Determined, Weak::No));
            }
        }

        // Forbid expanded shadowing to avoid time travel.
        if let Some(shadowed_glob) = shadowed_glob
            && shadowing == Shadowing::Restricted
            && finalize.stage == Stage::Early
            && binding.expansion != LocalExpnId::ROOT
            && binding.res() != shadowed_glob.res()
        {
            self.ambiguity_errors.push(AmbiguityError {
                kind: AmbiguityKind::GlobVsExpanded,
                ident,
                b1: binding,
                b2: shadowed_glob,
                warning: false,
                misc1: AmbiguityErrorMisc::None,
                misc2: AmbiguityErrorMisc::None,
            });
        }

        if shadowing == Shadowing::Unrestricted
            && binding.expansion != LocalExpnId::ROOT
            && let NameBindingKind::Import { import, .. } = binding.kind
            && matches!(import.kind, ImportKind::MacroExport)
        {
            self.macro_expanded_macro_export_errors.insert((path_span, binding.span));
        }

        // If we encounter a re-export for a type with private fields, it will not be able to
        // be constructed through this re-export. We track that case here to expand later
        // privacy errors with appropriate information.
        if let Res::Def(_, def_id) = binding.res() {
            let struct_ctor = match def_id.as_local() {
                Some(def_id) => self.struct_constructors.get(&def_id).cloned(),
                None => {
                    let ctor = self.cstore().ctor_untracked(def_id);
                    ctor.map(|(ctor_kind, ctor_def_id)| {
                        let ctor_res = Res::Def(
                            DefKind::Ctor(rustc_hir::def::CtorOf::Struct, ctor_kind),
                            ctor_def_id,
                        );
                        let ctor_vis = self.tcx.visibility(ctor_def_id);
                        let field_visibilities = self
                            .tcx
                            .associated_item_def_ids(def_id)
                            .iter()
                            .map(|field_id| self.tcx.visibility(field_id))
                            .collect();
                        (ctor_res, ctor_vis, field_visibilities)
                    })
                }
            };
            if let Some((_, _, fields)) = struct_ctor
                && fields.iter().any(|vis| !self.is_accessible_from(*vis, module))
            {
                self.inaccessible_ctor_reexport.insert(path_span, binding.span);
            }
        }

        self.record_use(ident, binding, used);
        return Ok(binding);
    }

    // Checks if a single import can define the `Ident` corresponding to `binding`.
    // This is used to check whether we can definitively accept a glob as a resolution.
    fn single_import_can_define_name<'r>(
        mut self: CmResolver<'r, 'ra, 'tcx>,
        resolution: &NameResolution<'ra>,
        binding: Option<NameBinding<'ra>>,
        ns: Namespace,
        ignore_import: Option<Import<'ra>>,
        ignore_binding: Option<NameBinding<'ra>>,
        parent_scope: &ParentScope<'ra>,
    ) -> bool {
        for single_import in &resolution.single_imports {
            if ignore_import == Some(*single_import) {
                continue;
            }
            if !self.is_accessible_from(single_import.vis, parent_scope.module) {
                continue;
            }
            if let Some(ignored) = ignore_binding
                && let NameBindingKind::Import { import, .. } = ignored.kind
                && import == *single_import
            {
                continue;
            }

            let Some(module) = single_import.imported_module.get() else {
                return true;
            };
            let ImportKind::Single { source, target, bindings, .. } = &single_import.kind else {
                unreachable!();
            };
            if source != target {
                if bindings.iter().all(|binding| binding.get().binding().is_none()) {
                    return true;
                } else if bindings[ns].get().binding().is_none() && binding.is_some() {
                    return true;
                }
            }

            match self.reborrow().resolve_ident_in_module(
                module,
                *source,
                ns,
                &single_import.parent_scope,
                None,
                ignore_binding,
                ignore_import,
            ) {
                Err((Determined, _)) => continue,
                Ok(binding)
                    if !self.is_accessible_from(binding.vis, single_import.parent_scope.module) =>
                {
                    continue;
                }
                Ok(_) | Err((Undetermined, _)) => {
                    return true;
                }
            }
        }

        false
    }

    /// Validate a local resolution (from ribs).
    #[instrument(level = "debug", skip(self, all_ribs))]
    fn validate_res_from_ribs(
        &mut self,
        rib_index: usize,
        rib_ident: Ident,
        mut res: Res,
        finalize: Option<Span>,
        original_rib_ident_def: Ident,
        all_ribs: &[Rib<'ra>],
    ) -> Res {
        debug!("validate_res_from_ribs({:?})", res);
        let ribs = &all_ribs[rib_index + 1..];

        // An invalid forward use of a generic parameter from a previous default
        // or in a const param ty.
        if let RibKind::ForwardGenericParamBan(reason) = all_ribs[rib_index].kind {
            if let Some(span) = finalize {
                let res_error = if rib_ident.name == kw::SelfUpper {
                    ResolutionError::ForwardDeclaredSelf(reason)
                } else {
                    ResolutionError::ForwardDeclaredGenericParam(rib_ident.name, reason)
                };
                self.report_error(span, res_error);
            }
            assert_eq!(res, Res::Err);
            return Res::Err;
        }

        match res {
            Res::Local(_) => {
                use ResolutionError::*;
                let mut res_err = None;

                for rib in ribs {
                    match rib.kind {
                        RibKind::Normal
                        | RibKind::Block(..)
                        | RibKind::FnOrCoroutine
                        | RibKind::Module(..)
                        | RibKind::MacroDefinition(..)
                        | RibKind::ForwardGenericParamBan(_) => {
                            // Nothing to do. Continue.
                        }
                        RibKind::Item(..) | RibKind::AssocItem => {
                            // This was an attempt to access an upvar inside a
                            // named function item. This is not allowed, so we
                            // report an error.
                            if let Some(span) = finalize {
                                // We don't immediately trigger a resolve error, because
                                // we want certain other resolution errors (namely those
                                // emitted for `ConstantItemRibKind` below) to take
                                // precedence.
                                res_err = Some((span, CannotCaptureDynamicEnvironmentInFnItem));
                            }
                        }
                        RibKind::ConstantItem(_, item) => {
                            // Still doesn't deal with upvars
                            if let Some(span) = finalize {
                                let (span, resolution_error) = match item {
                                    None if rib_ident.name == kw::SelfLower => {
                                        (span, LowercaseSelf)
                                    }
                                    None => {
                                        // If we have a `let name = expr;`, we have the span for
                                        // `name` and use that to see if it is followed by a type
                                        // specifier. If not, then we know we need to suggest
                                        // `const name: Ty = expr;`. This is a heuristic, it will
                                        // break down in the presence of macros.
                                        let sm = self.tcx.sess.source_map();
                                        let type_span = match sm.span_look_ahead(
                                            original_rib_ident_def.span,
                                            ":",
                                            None,
                                        ) {
                                            None => {
                                                Some(original_rib_ident_def.span.shrink_to_hi())
                                            }
                                            Some(_) => None,
                                        };
                                        (
                                            rib_ident.span,
                                            AttemptToUseNonConstantValueInConstant {
                                                ident: original_rib_ident_def,
                                                suggestion: "const",
                                                current: "let",
                                                type_span,
                                            },
                                        )
                                    }
                                    Some((ident, kind)) => (
                                        span,
                                        AttemptToUseNonConstantValueInConstant {
                                            ident,
                                            suggestion: "let",
                                            current: kind.as_str(),
                                            type_span: None,
                                        },
                                    ),
                                };
                                self.report_error(span, resolution_error);
                            }
                            return Res::Err;
                        }
                        RibKind::ConstParamTy => {
                            if let Some(span) = finalize {
                                self.report_error(
                                    span,
                                    ParamInTyOfConstParam { name: rib_ident.name },
                                );
                            }
                            return Res::Err;
                        }
                        RibKind::InlineAsmSym => {
                            if let Some(span) = finalize {
                                self.report_error(span, InvalidAsmSym);
                            }
                            return Res::Err;
                        }
                    }
                }
                if let Some((span, res_err)) = res_err {
                    self.report_error(span, res_err);
                    return Res::Err;
                }
            }
            Res::Def(DefKind::TyParam, _) | Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } => {
                for rib in ribs {
                    let (has_generic_params, def_kind) = match rib.kind {
                        RibKind::Normal
                        | RibKind::Block(..)
                        | RibKind::FnOrCoroutine
                        | RibKind::Module(..)
                        | RibKind::MacroDefinition(..)
                        | RibKind::InlineAsmSym
                        | RibKind::AssocItem
                        | RibKind::ForwardGenericParamBan(_) => {
                            // Nothing to do. Continue.
                            continue;
                        }

                        RibKind::ConstParamTy => {
                            if !self.tcx.features().generic_const_parameter_types() {
                                if let Some(span) = finalize {
                                    self.report_error(
                                        span,
                                        ResolutionError::ParamInTyOfConstParam {
                                            name: rib_ident.name,
                                        },
                                    );
                                }
                                return Res::Err;
                            } else {
                                continue;
                            }
                        }

                        RibKind::ConstantItem(trivial, _) => {
                            if let ConstantHasGenerics::No(cause) = trivial {
                                // HACK(min_const_generics): If we encounter `Self` in an anonymous
                                // constant we can't easily tell if it's generic at this stage, so
                                // we instead remember this and then enforce the self type to be
                                // concrete later on.
                                if let Res::SelfTyAlias {
                                    alias_to: def,
                                    forbid_generic: _,
                                    is_trait_impl,
                                } = res
                                {
                                    res = Res::SelfTyAlias {
                                        alias_to: def,
                                        forbid_generic: true,
                                        is_trait_impl,
                                    }
                                } else {
                                    if let Some(span) = finalize {
                                        let error = match cause {
                                            NoConstantGenericsReason::IsEnumDiscriminant => {
                                                ResolutionError::ParamInEnumDiscriminant {
                                                    name: rib_ident.name,
                                                    param_kind: ParamKindInEnumDiscriminant::Type,
                                                }
                                            }
                                            NoConstantGenericsReason::NonTrivialConstArg => {
                                                ResolutionError::ParamInNonTrivialAnonConst {
                                                    name: rib_ident.name,
                                                    param_kind:
                                                        ParamKindInNonTrivialAnonConst::Type,
                                                }
                                            }
                                        };
                                        let _: ErrorGuaranteed = self.report_error(span, error);
                                    }

                                    return Res::Err;
                                }
                            }

                            continue;
                        }

                        // This was an attempt to use a type parameter outside its scope.
                        RibKind::Item(has_generic_params, def_kind) => {
                            (has_generic_params, def_kind)
                        }
                    };

                    if let Some(span) = finalize {
                        self.report_error(
                            span,
                            ResolutionError::GenericParamsFromOuterItem(
                                res,
                                has_generic_params,
                                def_kind,
                            ),
                        );
                    }
                    return Res::Err;
                }
            }
            Res::Def(DefKind::ConstParam, _) => {
                for rib in ribs {
                    let (has_generic_params, def_kind) = match rib.kind {
                        RibKind::Normal
                        | RibKind::Block(..)
                        | RibKind::FnOrCoroutine
                        | RibKind::Module(..)
                        | RibKind::MacroDefinition(..)
                        | RibKind::InlineAsmSym
                        | RibKind::AssocItem
                        | RibKind::ForwardGenericParamBan(_) => continue,

                        RibKind::ConstParamTy => {
                            if !self.tcx.features().generic_const_parameter_types() {
                                if let Some(span) = finalize {
                                    self.report_error(
                                        span,
                                        ResolutionError::ParamInTyOfConstParam {
                                            name: rib_ident.name,
                                        },
                                    );
                                }
                                return Res::Err;
                            } else {
                                continue;
                            }
                        }

                        RibKind::ConstantItem(trivial, _) => {
                            if let ConstantHasGenerics::No(cause) = trivial {
                                if let Some(span) = finalize {
                                    let error = match cause {
                                        NoConstantGenericsReason::IsEnumDiscriminant => {
                                            ResolutionError::ParamInEnumDiscriminant {
                                                name: rib_ident.name,
                                                param_kind: ParamKindInEnumDiscriminant::Const,
                                            }
                                        }
                                        NoConstantGenericsReason::NonTrivialConstArg => {
                                            ResolutionError::ParamInNonTrivialAnonConst {
                                                name: rib_ident.name,
                                                param_kind: ParamKindInNonTrivialAnonConst::Const {
                                                    name: rib_ident.name,
                                                },
                                            }
                                        }
                                    };
                                    self.report_error(span, error);
                                }

                                return Res::Err;
                            }

                            continue;
                        }

                        RibKind::Item(has_generic_params, def_kind) => {
                            (has_generic_params, def_kind)
                        }
                    };

                    // This was an attempt to use a const parameter outside its scope.
                    if let Some(span) = finalize {
                        self.report_error(
                            span,
                            ResolutionError::GenericParamsFromOuterItem(
                                res,
                                has_generic_params,
                                def_kind,
                            ),
                        );
                    }
                    return Res::Err;
                }
            }
            _ => {}
        }

        res
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn maybe_resolve_path<'r>(
        self: CmResolver<'r, 'ra, 'tcx>,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'ra>,
        ignore_import: Option<Import<'ra>>,
    ) -> PathResult<'ra> {
        self.resolve_path_with_ribs(
            path,
            opt_ns,
            parent_scope,
            None,
            None,
            None,
            None,
            ignore_import,
        )
    }
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn resolve_path<'r>(
        self: CmResolver<'r, 'ra, 'tcx>,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'ra>,
        finalize: Option<Finalize>,
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
    ) -> PathResult<'ra> {
        self.resolve_path_with_ribs(
            path,
            opt_ns,
            parent_scope,
            None,
            finalize,
            None,
            ignore_binding,
            ignore_import,
        )
    }

    pub(crate) fn resolve_path_with_ribs<'r>(
        mut self: CmResolver<'r, 'ra, 'tcx>,
        path: &[Segment],
        opt_ns: Option<Namespace>, // `None` indicates a module path in import
        parent_scope: &ParentScope<'ra>,
        source: Option<PathSource<'_, '_, '_>>,
        finalize: Option<Finalize>,
        ribs: Option<&PerNS<Vec<Rib<'ra>>>>,
        ignore_binding: Option<NameBinding<'ra>>,
        ignore_import: Option<Import<'ra>>,
    ) -> PathResult<'ra> {
        let mut module = None;
        let mut module_had_parse_errors = false;
        let mut allow_super = true;
        let mut second_binding = None;

        // We'll provide more context to the privacy errors later, up to `len`.
        let privacy_errors_len = self.privacy_errors.len();
        fn record_segment_res<'r, 'ra, 'tcx>(
            mut this: CmResolver<'r, 'ra, 'tcx>,
            finalize: Option<Finalize>,
            res: Res,
            id: Option<NodeId>,
        ) {
            if finalize.is_some()
                && let Some(id) = id
                && !this.partial_res_map.contains_key(&id)
            {
                assert!(id != ast::DUMMY_NODE_ID, "Trying to resolve dummy id");
                this.get_mut().record_partial_res(id, PartialRes::new(res));
            }
        }

        for (segment_idx, &Segment { ident, id, .. }) in path.iter().enumerate() {
            debug!("resolve_path ident {} {:?} {:?}", segment_idx, ident, id);

            let is_last = segment_idx + 1 == path.len();
            let ns = if is_last { opt_ns.unwrap_or(TypeNS) } else { TypeNS };
            let name = ident.name;

            allow_super &= ns == TypeNS && (name == kw::SelfLower || name == kw::Super);

            if ns == TypeNS {
                if allow_super && name == kw::Super {
                    let mut ctxt = ident.span.ctxt().normalize_to_macros_2_0();
                    let self_module = match segment_idx {
                        0 => Some(self.resolve_self(&mut ctxt, parent_scope.module)),
                        _ => match module {
                            Some(ModuleOrUniformRoot::Module(module)) => Some(module),
                            _ => None,
                        },
                    };
                    if let Some(self_module) = self_module
                        && let Some(parent) = self_module.parent
                    {
                        module =
                            Some(ModuleOrUniformRoot::Module(self.resolve_self(&mut ctxt, parent)));
                        continue;
                    }
                    return PathResult::failed(
                        ident,
                        false,
                        finalize.is_some(),
                        module_had_parse_errors,
                        module,
                        || ("there are too many leading `super` keywords".to_string(), None),
                    );
                }
                if segment_idx == 0 {
                    if name == kw::SelfLower {
                        let mut ctxt = ident.span.ctxt().normalize_to_macros_2_0();
                        let self_mod = self.resolve_self(&mut ctxt, parent_scope.module);
                        if let Some(res) = self_mod.res() {
                            record_segment_res(self.reborrow(), finalize, res, id);
                        }
                        module = Some(ModuleOrUniformRoot::Module(self_mod));
                        continue;
                    }
                    if name == kw::PathRoot && ident.span.at_least_rust_2018() {
                        module = Some(ModuleOrUniformRoot::ExternPrelude);
                        continue;
                    }
                    if name == kw::PathRoot
                        && ident.span.is_rust_2015()
                        && self.tcx.sess.at_least_rust_2018()
                    {
                        // `::a::b` from 2015 macro on 2018 global edition
                        let crate_root = self.resolve_crate_root(ident);
                        module = Some(ModuleOrUniformRoot::ModuleAndExternPrelude(crate_root));
                        continue;
                    }
                    if name == kw::PathRoot || name == kw::Crate || name == kw::DollarCrate {
                        // `::a::b`, `crate::a::b` or `$crate::a::b`
                        let crate_root = self.resolve_crate_root(ident);
                        if let Some(res) = crate_root.res() {
                            record_segment_res(self.reborrow(), finalize, res, id);
                        }
                        module = Some(ModuleOrUniformRoot::Module(crate_root));
                        continue;
                    }
                }
            }

            // Report special messages for path segment keywords in wrong positions.
            if ident.is_path_segment_keyword() && segment_idx != 0 {
                return PathResult::failed(
                    ident,
                    false,
                    finalize.is_some(),
                    module_had_parse_errors,
                    module,
                    || {
                        let name_str = if name == kw::PathRoot {
                            "crate root".to_string()
                        } else {
                            format!("`{name}`")
                        };
                        let label = if segment_idx == 1 && path[0].ident.name == kw::PathRoot {
                            format!("global paths cannot start with {name_str}")
                        } else {
                            format!("{name_str} in paths can only be used in start position")
                        };
                        (label, None)
                    },
                );
            }

            let binding = if let Some(module) = module {
                self.reborrow()
                    .resolve_ident_in_module(
                        module,
                        ident,
                        ns,
                        parent_scope,
                        finalize,
                        ignore_binding,
                        ignore_import,
                    )
                    .map_err(|(determinacy, _)| determinacy)
            } else if let Some(ribs) = ribs
                && let Some(TypeNS | ValueNS) = opt_ns
            {
                assert!(ignore_import.is_none());
                match self.get_mut().resolve_ident_in_lexical_scope(
                    ident,
                    ns,
                    parent_scope,
                    finalize,
                    &ribs[ns],
                    ignore_binding,
                ) {
                    // we found a locally-imported or available item/module
                    Some(LexicalScopeBinding::Item(binding)) => Ok(binding),
                    // we found a local variable or type param
                    Some(LexicalScopeBinding::Res(res)) => {
                        record_segment_res(self.reborrow(), finalize, res, id);
                        return PathResult::NonModule(PartialRes::with_unresolved_segments(
                            res,
                            path.len() - 1,
                        ));
                    }
                    _ => Err(Determinacy::determined(finalize.is_some())),
                }
            } else {
                self.reborrow().resolve_ident_in_scope_set(
                    ident,
                    ScopeSet::All(ns),
                    parent_scope,
                    finalize,
                    finalize.is_some(),
                    ignore_binding,
                    ignore_import,
                )
            };

            match binding {
                Ok(binding) => {
                    if segment_idx == 1 {
                        second_binding = Some(binding);
                    }
                    let res = binding.res();

                    // Mark every privacy error in this path with the res to the last element. This allows us
                    // to detect the item the user cares about and either find an alternative import, or tell
                    // the user it is not accessible.
                    if finalize.is_some() {
                        for error in &mut self.get_mut().privacy_errors[privacy_errors_len..] {
                            error.outermost_res = Some((res, ident));
                            error.source = match source {
                                Some(PathSource::Struct(Some(expr)))
                                | Some(PathSource::Expr(Some(expr))) => Some(expr.clone()),
                                _ => None,
                            };
                        }
                    }

                    let maybe_assoc = opt_ns != Some(MacroNS) && PathSource::Type.is_expected(res);
                    if let Some(def_id) = binding.res().module_like_def_id() {
                        if self.mods_with_parse_errors.contains(&def_id) {
                            module_had_parse_errors = true;
                        }
                        module = Some(ModuleOrUniformRoot::Module(self.expect_module(def_id)));
                        record_segment_res(self.reborrow(), finalize, res, id);
                    } else if res == Res::ToolMod && !is_last && opt_ns.is_some() {
                        if binding.is_import() {
                            self.dcx().emit_err(errors::ToolModuleImported {
                                span: ident.span,
                                import: binding.span,
                            });
                        }
                        let res = Res::NonMacroAttr(NonMacroAttrKind::Tool);
                        return PathResult::NonModule(PartialRes::new(res));
                    } else if res == Res::Err {
                        return PathResult::NonModule(PartialRes::new(Res::Err));
                    } else if opt_ns.is_some() && (is_last || maybe_assoc) {
                        if let Some(finalize) = finalize {
                            self.get_mut().lint_if_path_starts_with_module(
                                finalize,
                                path,
                                second_binding,
                            );
                        }
                        record_segment_res(self.reborrow(), finalize, res, id);
                        return PathResult::NonModule(PartialRes::with_unresolved_segments(
                            res,
                            path.len() - segment_idx - 1,
                        ));
                    } else {
                        return PathResult::failed(
                            ident,
                            is_last,
                            finalize.is_some(),
                            module_had_parse_errors,
                            module,
                            || {
                                let label = format!(
                                    "`{ident}` is {} {}, not a module",
                                    res.article(),
                                    res.descr()
                                );
                                (label, None)
                            },
                        );
                    }
                }
                Err(Undetermined) => return PathResult::Indeterminate,
                Err(Determined) => {
                    if let Some(ModuleOrUniformRoot::Module(module)) = module
                        && opt_ns.is_some()
                        && !module.is_normal()
                    {
                        return PathResult::NonModule(PartialRes::with_unresolved_segments(
                            module.res().unwrap(),
                            path.len() - segment_idx,
                        ));
                    }

                    let mut this = self.reborrow();
                    return PathResult::failed(
                        ident,
                        is_last,
                        finalize.is_some(),
                        module_had_parse_errors,
                        module,
                        || {
                            this.get_mut().report_path_resolution_error(
                                path,
                                opt_ns,
                                parent_scope,
                                ribs,
                                ignore_binding,
                                ignore_import,
                                module,
                                segment_idx,
                                ident,
                            )
                        },
                    );
                }
            }
        }

        if let Some(finalize) = finalize {
            self.get_mut().lint_if_path_starts_with_module(finalize, path, second_binding);
        }

        PathResult::Module(match module {
            Some(module) => module,
            None if path.is_empty() => ModuleOrUniformRoot::CurrentScope,
            _ => bug!("resolve_path: non-empty path `{:?}` has no module", path),
        })
    }
}
