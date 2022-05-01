use crate::{ParentScope, Resolver};

use rustc_ast::ptr::P;
use rustc_ast::visit::{self, AssocCtxt, BoundKind, FnKind, Visitor};
use rustc_ast::*;
use rustc_ast_lowering::{LifetimeRes, ResolverAstLowering};
use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::def::DefKind;
use rustc_hir::definitions::DefPathData;
use rustc_index::vec::Idx;
use rustc_middle::ty::DefIdTree;
use rustc_session::lint;
use rustc_span::symbol::{kw, Ident};
use rustc_span::{BytePos, Span};

use std::mem::replace;
use tracing::debug;

use super::*;

#[derive(Copy, Clone, Debug)]
pub(super) enum LifetimeRibKind {
    /// This rib acts as a barrier to forbid reference to lifetimes of a parent item.
    Item,

    /// This rib declares generic parameters.
    Generics { parent: NodeId, span: Span, kind: LifetimeBinderKind },

    /// FIXME(const_generics): This patches over an ICE caused by non-'static lifetimes in const
    /// generics. We are disallowing this until we can decide on how we want to handle non-'static
    /// lifetimes in const generics. See issue #74052 for discussion.
    ConstGeneric,

    /// Non-static lifetimes are prohibited in anonymous constants under `min_const_generics`.
    /// This function will emit an error if `generic_const_exprs` is not enabled, the body identified by
    /// `body_id` is an anonymous constant and `lifetime_ref` is non-static.
    AnonConst,

    /// For **Modern** cases, create a new anonymous region parameter
    /// and reference that.
    ///
    /// For **Dyn Bound** cases, pass responsibility to
    /// `resolve_lifetime` code.
    ///
    /// For **Deprecated** cases, report an error.
    AnonymousCreateParameter(NodeId),

    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    AnonymousReportError,

    /// Pass responsibility to `resolve_lifetime` code for all cases.
    AnonymousPassThrough(NodeId),
}

#[derive(Copy, Clone, Debug)]
pub(super) enum LifetimeBinderKind {
    BareFnType,
    PolyTrait,
    WhereBound,
    Item,
    Function,
    ImplBlock,
}

impl LifetimeBinderKind {
    pub fn descr(self) -> &'static str {
        use LifetimeBinderKind::*;
        match self {
            BareFnType => "type",
            PolyTrait => "bound",
            WhereBound => "bound",
            Item => "item",
            ImplBlock => "impl block",
            Function => "function",
        }
    }
}

#[derive(Debug)]
pub(super) struct LifetimeRib {
    pub kind: LifetimeRibKind,
    // We need to preserve insertion order for async fns.
    bindings: FxIndexMap<Ident, (NodeId, LifetimeRes)>,
}

impl LifetimeRib {
    fn new(kind: LifetimeRibKind) -> LifetimeRib {
        LifetimeRib { bindings: Default::default(), kind }
    }
}

pub(super) struct LifetimeResolutionVisitor<'a, 'b> {
    pub r: &'b mut Resolver<'a>,

    /// The module that represents the current item scope.
    parent_scope: ParentScope<'a>,

    /// The current set of local scopes for lifetimes.
    pub lifetime_ribs: Vec<LifetimeRib>,
}

/// Walks the whole crate in DFS order, visiting each item, resolving names as it goes.
impl<'a: 'ast, 'ast> Visitor<'ast> for LifetimeResolutionVisitor<'a, '_> {
    fn visit_attribute(&mut self, _: &'ast Attribute) {
        // We do not want to resolve expressions that appear in attributes,
        // as they do not correspond to actual code.
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        self.resolve_lifetime(lifetime)
    }

    // Items
    fn visit_item(&mut self, item: &'ast Item) {
        self.with_lifetime_rib(LifetimeRibKind::Item, |this| this.resolve_item(item));
    }
    fn visit_assoc_item(&mut self, item: &'ast AssocItem, ctxt: visit::AssocCtxt) {
        let walk_assoc_item =
            |this: &mut Self, generics: &Generics, kind, item: &'ast AssocItem| {
                this.with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics { parent: item.id, span: generics.span, kind },
                    |this| visit::walk_assoc_item(this, item, ctxt),
                );
            };

        match &item.kind {
            AssocItemKind::Const(..) => visit::walk_assoc_item(self, item, ctxt),
            AssocItemKind::Fn(box Fn { generics, .. }) => {
                walk_assoc_item(self, generics, LifetimeBinderKind::Function, item);
            }
            AssocItemKind::TyAlias(box TyAlias { generics, .. }) => {
                walk_assoc_item(self, generics, LifetimeBinderKind::Item, item);
            }
            AssocItemKind::MacCall(_) => panic!("unexpanded macro in resolve!"),
        }
    }
    fn visit_foreign_item(&mut self, foreign_item: &'ast ForeignItem) {
        self.with_lifetime_rib(LifetimeRibKind::Item, |this| match foreign_item.kind {
            ForeignItemKind::TyAlias(box TyAlias { ref generics, .. }) => this
                .with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics {
                        parent: foreign_item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_foreign_item(this, foreign_item),
                ),
            ForeignItemKind::Fn(box Fn { ref generics, .. }) => this.with_generic_param_rib(
                &generics.params,
                LifetimeRibKind::Generics {
                    parent: foreign_item.id,
                    kind: LifetimeBinderKind::Function,
                    span: generics.span,
                },
                |this| visit::walk_foreign_item(this, foreign_item),
            ),
            ForeignItemKind::Static(..) => visit::walk_foreign_item(this, foreign_item),
            ForeignItemKind::MacCall(..) => panic!("unexpanded macro in resolve!"),
        });
    }

    // Expressions
    fn visit_block(&mut self, block: &'ast Block) {
        debug!("lifetime resolution: entering block");

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.parent_scope.module;
        let anonymous_module = self.r.block_map.get(&block.id).copied();

        if let Some(anonymous_module) = anonymous_module {
            debug!("(resolving block) found anonymous module, moving down");
            self.parent_scope.module = anonymous_module;
        }

        // Descend into the block.
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }

        // Move back up.
        self.parent_scope.module = orig_module;
        debug!("lifetime resolution: leaving block");
    }
    fn visit_anon_const(&mut self, constant: &'ast AnonConst) {
        self.with_lifetime_rib(LifetimeRibKind::AnonConst, |this| {
            visit::walk_anon_const(this, constant)
        })
    }
    fn visit_expr(&mut self, expr: &'ast Expr) {
        match expr.kind {
            ExprKind::Path(_, ref path) => {
                self.resolve_elided_lifetimes_in_path(expr.id, path, false);
            }
            ExprKind::Struct(ref se) => {
                self.resolve_elided_lifetimes_in_path(expr.id, &se.path, false);
            }
            ExprKind::ConstBlock(ref block) => {
                // Do not use `visit_anon_const`: we want to keep the current rib.
                self.visit_expr(&block.value);
                return;
            }
            _ => {}
        }
        visit::walk_expr(self, expr);
    }
    fn visit_pat(&mut self, pat: &Pat) {
        match pat.kind {
            PatKind::TupleStruct(_, ref path, _)
            | PatKind::Path(_, ref path)
            | PatKind::Struct(_, ref path, ..) => {
                self.resolve_elided_lifetimes_in_path(pat.id, path, false);
            }
            _ => {}
        }
        visit::walk_pat(self, pat);
    }
    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) {
        // This is similar to the code for AnonConst.
        self.resolve_elided_lifetimes_in_path(sym.id, &sym.path, false);
        visit::walk_inline_asm_sym(self, sym);
    }

    // Types
    fn visit_ty(&mut self, ty: &'ast Ty) {
        match ty.kind {
            TyKind::Rptr(None, _) => {
                // Elided lifetime in reference: we resolve as if there was some lifetime `'_` with
                // NodeId `ty.id`.
                let span = self.r.session.source_map().next_point(ty.span.shrink_to_lo());
                self.resolve_elided_lifetime(ty.id, span);
            }
            TyKind::Path(_, ref path) => {
                self.resolve_elided_lifetimes_in_path(ty.id, path, true);
            }
            TyKind::ImplicitSelf => {}
            TyKind::BareFn(ref bare_fn) => {
                let span = if bare_fn.generic_params.is_empty() {
                    ty.span.shrink_to_lo()
                } else {
                    ty.span
                };
                self.with_generic_param_rib(
                    &bare_fn.generic_params,
                    LifetimeRibKind::Generics {
                        parent: ty.id,
                        kind: LifetimeBinderKind::BareFnType,
                        span,
                    },
                    |this| {
                        this.with_lifetime_rib(
                            LifetimeRibKind::AnonymousPassThrough(ty.id),
                            |this| {
                                walk_list!(this, visit_generic_param, &bare_fn.generic_params);
                                visit::walk_fn_decl(this, &bare_fn.decl);
                            },
                        );
                    },
                );
                return;
            }
            _ => (),
        }
        visit::walk_ty(self, ty);
    }
    fn visit_poly_trait_ref(&mut self, tref: &'ast PolyTraitRef, tbm: &'ast TraitBoundModifier) {
        let span =
            if tref.bound_generic_params.is_empty() { tref.span.shrink_to_lo() } else { tref.span };
        self.with_generic_param_rib(
            &tref.bound_generic_params,
            LifetimeRibKind::Generics {
                parent: tref.trait_ref.ref_id,
                kind: LifetimeBinderKind::PolyTrait,
                span,
            },
            |this| {
                this.resolve_elided_lifetimes_in_path(
                    tref.trait_ref.ref_id,
                    &tref.trait_ref.path,
                    true,
                );
                visit::walk_poly_trait_ref(this, tref, tbm);
            },
        );
    }

    fn visit_fn(&mut self, fn_kind: FnKind<'ast>, _: Span, fn_id: NodeId) {
        debug!("lifetime resolution: entering function");
        let declaration = fn_kind.decl();

        let async_node_id = fn_kind.header().and_then(|h| h.asyncness.opt_return_id());

        if let FnKind::Fn(_, _, _, _, generics, _) = fn_kind {
            self.visit_generics(generics);
        }

        if let Some(async_node_id) = async_node_id {
            // In `async fn`, argument-position elided lifetimes
            // must be transformed into fresh generic parameters so that
            // they can be applied to the opaque `impl Trait` return type.
            self.with_lifetime_rib(LifetimeRibKind::AnonymousCreateParameter(fn_id), |this| {
                // Add each argument to the rib.
                for Param { pat, ty, .. } in &declaration.inputs {
                    this.visit_pat(pat);
                    this.visit_ty(ty);
                }
            });

            // Construct the list of in-scope lifetime parameters for async lowering.
            // We include all lifetime parameters, either named or "Fresh".
            // The order of those parameters does not matter, as long as it is
            // deterministic.
            let mut extra_lifetime_params =
                self.r.extra_lifetime_params_map.get(&fn_id).cloned().unwrap_or_default();
            for rib in self.lifetime_ribs.iter().rev() {
                extra_lifetime_params.extend(
                    rib.bindings.iter().map(|(&ident, &(node_id, res))| (ident, node_id, res)),
                );
                match rib.kind {
                    LifetimeRibKind::Item => break,
                    LifetimeRibKind::AnonymousCreateParameter(id) => {
                        if let Some(earlier_fresh) = self.r.extra_lifetime_params_map.get(&id) {
                            extra_lifetime_params.extend(earlier_fresh);
                        }
                    }
                    _ => {}
                }
            }
            self.r.extra_lifetime_params_map.insert(async_node_id, extra_lifetime_params);

            self.with_lifetime_rib(LifetimeRibKind::AnonymousPassThrough(async_node_id), |this| {
                visit::walk_fn_ret_ty(this, &declaration.output)
            });
        } else {
            self.with_lifetime_rib(LifetimeRibKind::AnonymousPassThrough(fn_id), |this| {
                // Add each argument to the rib.
                for Param { pat, ty, .. } in &declaration.inputs {
                    this.visit_pat(pat);
                    this.visit_ty(ty);
                }

                visit::walk_fn_ret_ty(this, &declaration.output);
            });
        };

        // Ignore errors in function bodies if this is rustdoc
        // Be sure not to set this until the function signature has been resolved.
        // Resolve the function body, potentially inside the body of an async closure
        self.with_lifetime_rib(
            LifetimeRibKind::AnonymousPassThrough(fn_id),
            |this| match fn_kind {
                FnKind::Fn(.., body) => walk_list!(this, visit_block, body),
                FnKind::Closure(_, body) => this.visit_expr(body),
            },
        );

        debug!("lifetime resolution: leaving function");
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'ast PathSegment) {
        if let Some(ref args) = path_segment.args {
            match &**args {
                GenericArgs::AngleBracketed(..) => visit::walk_generic_args(self, path_span, args),
                GenericArgs::Parenthesized(..) => self.with_lifetime_rib(
                    LifetimeRibKind::AnonymousPassThrough(path_segment.id),
                    |this| visit::walk_generic_args(this, path_span, args),
                ),
            }
        }
    }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| match p {
            WherePredicate::BoundPredicate(WhereBoundPredicate {
                ref bounded_ty,
                ref bound_generic_params,
                span: predicate_span,
                ..
            }) => {
                let span = if bound_generic_params.is_empty() {
                    predicate_span.shrink_to_lo()
                } else {
                    *predicate_span
                };
                this.with_generic_param_rib(
                    &bound_generic_params,
                    LifetimeRibKind::Generics {
                        parent: bounded_ty.id,
                        kind: LifetimeBinderKind::WhereBound,
                        span,
                    },
                    |this| visit::walk_where_predicate(this, p),
                );
            }
            _ => visit::walk_where_predicate(this, p),
        });
    }
    fn visit_generic_param(&mut self, param: &'ast GenericParam) {
        self.with_lifetime_rib(LifetimeRibKind::AnonymousReportError, |this| {
            match param.kind {
                GenericParamKind::Lifetime | GenericParamKind::Type { .. } => {
                    visit::walk_generic_param(this, param)
                }
                GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                    // Const parameters can't have param bounds.
                    assert!(param.bounds.is_empty());
                    this.with_lifetime_rib(LifetimeRibKind::ConstGeneric, |this| {
                        this.visit_ty(ty);

                        if let Some(ref expr) = default {
                            // Do not use `visit_anon_const`: we want to keep the
                            // `ConstGeneric` rib.
                            this.visit_expr(&expr.value)
                        }
                    });
                }
            }
        })
    }
}

impl<'a: 'ast, 'b, 'ast> LifetimeResolutionVisitor<'a, 'b> {
    pub fn new(resolver: &'b mut Resolver<'a>) -> LifetimeResolutionVisitor<'a, 'b> {
        // During late resolution we only track the module component of the parent scope,
        // although it may be useful to track other components as well for diagnostics.
        let graph_root = resolver.graph_root;
        let parent_scope = ParentScope::module(graph_root, resolver);
        LifetimeResolutionVisitor { r: resolver, parent_scope, lifetime_ribs: Vec::new() }
    }

    fn with_scope<T>(&mut self, id: NodeId, f: impl FnOnce(&mut Self) -> T) -> T {
        if let Some(module) = self.r.get_module(self.r.local_def_id(id).to_def_id()) {
            // Move down in the graph.
            let orig_module = replace(&mut self.parent_scope.module, module);
            let ret = f(self);
            self.parent_scope.module = orig_module;
            ret
        } else {
            f(self)
        }
    }

    #[tracing::instrument(level = "debug", skip(self, work))]
    fn with_lifetime_rib<T>(
        &mut self,
        kind: LifetimeRibKind,
        work: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.lifetime_ribs.push(LifetimeRib::new(kind));
        let ret = work(self);
        self.lifetime_ribs.pop();
        ret
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn resolve_lifetime(&mut self, lifetime: &'ast Lifetime) {
        let ident = lifetime.ident;

        if ident.name == kw::StaticLifetime {
            self.record_lifetime_res(lifetime.id, LifetimeRes::Static);
            return;
        }

        if ident.name == kw::UnderscoreLifetime {
            return self.resolve_anonymous_lifetime(lifetime, false);
        }

        let mut indices = (0..self.lifetime_ribs.len()).rev();
        for i in &mut indices {
            let rib = &self.lifetime_ribs[i];
            let normalized_ident = ident.normalize_to_macros_2_0();
            if let Some(&(_, region)) = rib.bindings.get(&normalized_ident) {
                self.record_lifetime_res(lifetime.id, region);
                return;
            }

            match rib.kind {
                LifetimeRibKind::Item => break,
                LifetimeRibKind::ConstGeneric => {
                    self.emit_non_static_lt_in_const_generic_error(lifetime);
                    self.r.lifetimes_res_map.insert(lifetime.id, LifetimeRes::Error);
                    return;
                }
                LifetimeRibKind::AnonConst => {
                    self.maybe_emit_forbidden_non_static_lifetime_error(lifetime);
                    self.r.lifetimes_res_map.insert(lifetime.id, LifetimeRes::Error);
                    return;
                }
                _ => {}
            }
        }

        let mut outer_res = None;
        for i in indices {
            let rib = &self.lifetime_ribs[i];
            let normalized_ident = ident.normalize_to_macros_2_0();
            if let Some((&outer, _)) = rib.bindings.get_key_value(&normalized_ident) {
                outer_res = Some(outer);
                break;
            }
        }

        self.emit_undeclared_lifetime_error(lifetime, outer_res);
        self.record_lifetime_res(lifetime.id, LifetimeRes::Error);
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn resolve_anonymous_lifetime(&mut self, lifetime: &Lifetime, elided: bool) {
        debug_assert_eq!(lifetime.ident.name, kw::UnderscoreLifetime);

        for i in (0..self.lifetime_ribs.len()).rev() {
            let rib = &mut self.lifetime_ribs[i];
            match rib.kind {
                LifetimeRibKind::AnonymousCreateParameter(item_node_id) => {
                    self.create_fresh_lifetime(lifetime.id, lifetime.ident, item_node_id);
                    return;
                }
                LifetimeRibKind::AnonymousReportError => {
                    let (msg, note) = if elided {
                        (
                            "`&` without an explicit lifetime name cannot be used here",
                            "explicit lifetime name needed here",
                        )
                    } else {
                        ("`'_` cannot be used here", "`'_` is a reserved lifetime name")
                    };
                    rustc_errors::struct_span_err!(
                        self.r.session,
                        lifetime.ident.span,
                        E0637,
                        "{}",
                        msg,
                    )
                    .span_label(lifetime.ident.span, note)
                    .emit();

                    self.record_lifetime_res(lifetime.id, LifetimeRes::Error);
                    return;
                }
                LifetimeRibKind::AnonymousPassThrough(node_id) => {
                    self.record_lifetime_res(
                        lifetime.id,
                        LifetimeRes::Anonymous { binder: node_id, elided },
                    );
                    return;
                }
                LifetimeRibKind::Item => break,
                _ => {}
            }
        }
        // This resolution is wrong, it passes the work to HIR lifetime resolution.
        // We cannot use `LifetimeRes::Error` because we do not emit a diagnostic.
        self.record_lifetime_res(
            lifetime.id,
            LifetimeRes::Anonymous { binder: DUMMY_NODE_ID, elided },
        );
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn resolve_elided_lifetime(&mut self, anchor_id: NodeId, span: Span) {
        let id = self.r.next_node_id();
        self.record_lifetime_res(
            anchor_id,
            LifetimeRes::ElidedAnchor { start: id, end: NodeId::from_u32(id.as_u32() + 1) },
        );

        let lt = Lifetime { id, ident: Ident::new(kw::UnderscoreLifetime, span) };
        self.resolve_anonymous_lifetime(&lt, true);
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn create_fresh_lifetime(&mut self, id: NodeId, ident: Ident, item_node_id: NodeId) {
        debug_assert_eq!(ident.name, kw::UnderscoreLifetime);
        debug!(?ident.span);
        let item_def_id = self.r.local_def_id(item_node_id);
        let def_node_id = self.r.next_node_id();
        let def_id = self.r.create_def(
            item_def_id,
            def_node_id,
            DefPathData::LifetimeNs(kw::UnderscoreLifetime),
            self.parent_scope.expansion.to_expn_id(),
            ident.span,
        );
        debug!(?def_id);

        let region = LifetimeRes::Fresh { param: def_id, binder: item_node_id };
        self.record_lifetime_res(id, region);
        self.r.extra_lifetime_params_map.entry(item_node_id).or_insert_with(Vec::new).push((
            ident,
            def_node_id,
            region,
        ));
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn resolve_elided_lifetimes_in_path(&mut self, path_id: NodeId, path: &Path, missing: bool) {
        let path_span = path.span;
        let Some(&partial_res) = self.r.partial_res_map.get(&path_id) else { return };
        let proj_start = path.segments.len() - partial_res.unresolved_segments();
        for (i, segment) in path.segments.iter().enumerate() {
            let (args_span, has_lifetime_args) = if let Some(args) = segment.args.as_deref() {
                match args {
                    GenericArgs::AngleBracketed(args) => {
                        let found_lifetimes = args.args.iter().any(|arg| {
                            matches!(arg, AngleBracketedArg::Arg(GenericArg::Lifetime(_)))
                        });
                        (args.span, found_lifetimes)
                    }
                    GenericArgs::Parenthesized(args) => (args.span, true),
                }
            } else {
                (rustc_span::DUMMY_SP, false)
            };

            if has_lifetime_args {
                continue;
            }

            // Figure out if this is a type/trait segment,
            // which may need lifetime elision performed.
            let type_def_id = match partial_res.base_res() {
                Res::Def(DefKind::AssocTy, def_id) if i + 2 == proj_start => self.r.parent(def_id),
                Res::Def(DefKind::Variant, def_id) if i + 1 == proj_start => self.r.parent(def_id),
                Res::Def(DefKind::Struct, def_id)
                | Res::Def(DefKind::Union, def_id)
                | Res::Def(DefKind::Enum, def_id)
                | Res::Def(DefKind::TyAlias, def_id)
                | Res::Def(DefKind::Trait, def_id)
                    if i + 1 == proj_start =>
                {
                    def_id
                }
                _ => continue,
            };

            let expected_lifetimes = self.r.item_generics_num_lifetimes(type_def_id);
            if expected_lifetimes == 0 {
                continue;
            }

            let node_ids = self.r.next_node_ids(expected_lifetimes);
            self.record_lifetime_res(
                segment.id,
                LifetimeRes::ElidedAnchor { start: node_ids.start, end: node_ids.end },
            );

            if !missing {
                let res = LifetimeRes::Anonymous { binder: DUMMY_NODE_ID, elided: true };
                for i in 0..expected_lifetimes {
                    let id = node_ids.start.plus(i);
                    self.record_lifetime_res(id, res);
                }
                continue;
            }

            let mut res = LifetimeRes::Error;
            for rib in self.lifetime_ribs.iter().rev() {
                match rib.kind {
                    // In create-parameter mode we error here because we don't want to support
                    // deprecated impl elision in new features like impl elision and `async fn`,
                    // both of which work using the `CreateParameter` mode:
                    //
                    //     impl Foo for std::cell::Ref<u32> // note lack of '_
                    //     async fn foo(_: std::cell::Ref<u32>) { ... }
                    LifetimeRibKind::AnonymousCreateParameter(_) => {
                        break;
                    }
                    // `PassThrough` is the normal case.
                    // `new_error_lifetime`, which would usually be used in the case of `ReportError`,
                    // is unsuitable here, as these can occur from missing lifetime parameters in a
                    // `PathSegment`, for which there is no associated `'_` or `&T` with no explicit
                    // lifetime. Instead, we simply create an implicit lifetime, which will be checked
                    // later, at which point a suitable error will be emitted.
                    LifetimeRibKind::AnonymousPassThrough(binder) => {
                        res = LifetimeRes::Anonymous { binder, elided: true };
                        break;
                    }
                    LifetimeRibKind::AnonymousReportError | LifetimeRibKind::Item => {
                        // FIXME(cjgillot) This resolution is wrong, but this does not matter
                        // since these cases are erroneous anyway.  Lifetime resolution should
                        // emit a "missing lifetime specifier" diagnostic.
                        res = LifetimeRes::Anonymous { binder: DUMMY_NODE_ID, elided: true };
                        break;
                    }
                    _ => {}
                }
            }

            for i in 0..expected_lifetimes {
                let id = node_ids.start.plus(i);
                self.record_lifetime_res(id, res);
            }

            let has_generic_args = segment.args.is_some();
            let elided_lifetime_span = if has_generic_args {
                // If there are brackets, but not generic arguments, then use the opening bracket
                args_span.with_hi(args_span.lo() + BytePos(1))
            } else {
                // If there are no brackets, use the identifier span.
                // HACK: we use find_ancestor_inside to properly suggest elided spans in paths
                // originating from macros, since the segment's span might be from a macro arg.
                segment.ident.span.find_ancestor_inside(path_span).unwrap_or(path_span)
            };
            if let LifetimeRes::Error = res {
                let sess = self.r.session;
                let mut err = rustc_errors::struct_span_err!(
                    sess,
                    path_span,
                    E0726,
                    "implicit elided lifetime not allowed here"
                );
                rustc_errors::add_elided_lifetime_in_path_suggestion(
                    sess.source_map(),
                    &mut err,
                    expected_lifetimes,
                    path_span,
                    !has_generic_args,
                    elided_lifetime_span,
                );
                err.note("assuming a `'static` lifetime...");
                err.emit();
            } else {
                self.r.lint_buffer.buffer_lint_with_diagnostic(
                    lint::builtin::ELIDED_LIFETIMES_IN_PATHS,
                    segment.id,
                    elided_lifetime_span,
                    "hidden lifetime parameters in types are deprecated",
                    lint::BuiltinLintDiagnostics::ElidedLifetimesInPaths(
                        expected_lifetimes,
                        path_span,
                        !has_generic_args,
                        elided_lifetime_span,
                    ),
                );
            }
        }
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn record_lifetime_res(&mut self, id: NodeId, res: LifetimeRes) {
        if let Some(prev_res) = self.r.lifetimes_res_map.insert(id, res) {
            panic!(
                "lifetime {:?} resolved multiple times ({:?} before, {:?} now)",
                id, prev_res, res
            )
        }
    }

    fn resolve_adt(&mut self, item: &'ast Item, generics: &'ast Generics) {
        debug!("resolve_adt");
        self.with_generic_param_rib(
            &generics.params,
            LifetimeRibKind::Generics {
                parent: item.id,
                kind: LifetimeBinderKind::Item,
                span: generics.span,
            },
            |this| visit::walk_item(this, item),
        );
    }

    fn resolve_item(&mut self, item: &'ast Item) {
        let name = item.ident.name;
        debug!("(resolving item) resolving {} ({:?})", name, item.kind);

        match item.kind {
            ItemKind::TyAlias(box TyAlias { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics {
                        parent: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
            }

            ItemKind::Fn(box Fn { ref generics, .. }) => {
                self.with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics {
                        parent: item.id,
                        kind: LifetimeBinderKind::Function,
                        span: generics.span,
                    },
                    |this| visit::walk_item(this, item),
                );
            }

            ItemKind::Enum(_, ref generics)
            | ItemKind::Struct(_, ref generics)
            | ItemKind::Union(_, ref generics) => {
                self.resolve_adt(item, generics);
            }

            ItemKind::Impl(box Impl {
                ref generics,
                ref of_trait,
                ref self_ty,
                items: ref impl_items,
                ..
            }) => {
                self.resolve_implementation(generics, of_trait, &self_ty, item.id, impl_items);
            }

            ItemKind::Trait(box Trait { ref generics, ref bounds, ref items, .. }) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics {
                        parent: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_param_bound, bounds, BoundKind::SuperTraits);
                        walk_list!(this, visit_assoc_item, items, AssocCtxt::Trait);
                    },
                );
            }

            ItemKind::TraitAlias(ref generics, ref bounds) => {
                // Create a new rib for the trait-wide type parameters.
                self.with_generic_param_rib(
                    &generics.params,
                    LifetimeRibKind::Generics {
                        parent: item.id,
                        kind: LifetimeBinderKind::Item,
                        span: generics.span,
                    },
                    |this| {
                        this.visit_generics(generics);
                        walk_list!(this, visit_param_bound, bounds, BoundKind::Bound);
                    },
                );
            }

            ItemKind::Mod(..) | ItemKind::ForeignMod(_) => {
                self.with_scope(item.id, |this| {
                    visit::walk_item(this, item);
                });
            }

            ItemKind::Static(..) | ItemKind::Const(..) => {
                self.with_lifetime_rib(LifetimeRibKind::Item, |this| visit::walk_item(this, item));
            }

            ItemKind::Use(..) => {
                // do nothing, this cannot possibly reference lifetimes
            }

            ItemKind::ExternCrate(..) | ItemKind::MacroDef(..) => {
                // do nothing, these are just around to be encoded
            }

            ItemKind::GlobalAsm(_) => {
                visit::walk_item(self, item);
            }

            ItemKind::MacCall(_) => panic!("unexpanded macro in resolve!"),
        }
    }

    fn with_generic_param_rib<'c, F>(
        &'c mut self,
        params: &'c Vec<GenericParam>,
        lifetime_kind: LifetimeRibKind,
        f: F,
    ) where
        F: FnOnce(&mut Self),
    {
        debug!("with_generic_param_rib");

        self.with_lifetime_rib(lifetime_kind, |this| {
            for param in params {
                let ident = param.ident.normalize_to_macros_2_0();
                debug!("with_generic_param_rib: {}", param.id);

                let GenericParamKind::Lifetime = param.kind else { continue };

                if param.ident.name == kw::UnderscoreLifetime {
                    rustc_errors::struct_span_err!(
                        this.r.session,
                        param.ident.span,
                        E0637,
                        "`'_` cannot be used here"
                    )
                    .span_label(param.ident.span, "`'_` is a reserved lifetime name")
                    .emit();
                    continue;
                }

                if param.ident.name == kw::StaticLifetime {
                    rustc_errors::struct_span_err!(
                        this.r.session,
                        param.ident.span,
                        E0262,
                        "invalid lifetime parameter name: `{}`",
                        param.ident,
                    )
                    .span_label(param.ident.span, "'static is a reserved lifetime name")
                    .emit();
                    continue;
                }

                let def_id = this.r.local_def_id(param.id);

                // Plain insert (no renaming).
                match param.kind {
                    GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => continue,
                    GenericParamKind::Lifetime => {}
                };

                let LifetimeRibKind::Generics { parent, .. } = lifetime_kind else { panic!() };
                let res = LifetimeRes::Param { param: def_id, binder: parent };
                this.record_lifetime_res(param.id, res);
                this.lifetime_ribs.last_mut().unwrap().bindings.insert(ident, (param.id, res));
            }

            f(this);
        })
    }

    fn resolve_implementation(
        &mut self,
        generics: &'ast Generics,
        opt_trait_reference: &'ast Option<TraitRef>,
        self_type: &'ast Ty,
        item_id: NodeId,
        impl_items: &'ast [P<AssocItem>],
    ) {
        debug!("resolve_implementation");
        self.with_generic_param_rib(
            &generics.params,
            LifetimeRibKind::Generics {
                span: generics.span,
                parent: item_id,
                kind: LifetimeBinderKind::ImplBlock,
            },
            |this| {
                this.with_lifetime_rib(
                    LifetimeRibKind::AnonymousCreateParameter(item_id),
                    |this| {
                        // Resolve the trait reference, if necessary.
                        if let Some(trait_ref) = opt_trait_reference.as_ref() {
                            this.resolve_elided_lifetimes_in_path(
                                trait_ref.ref_id,
                                &trait_ref.path,
                                true,
                            );
                            // Resolve type arguments in the trait path.
                            visit::walk_trait_ref(this, trait_ref);
                        }

                        // Resolve the self type.
                        this.visit_ty(self_type);
                        // Resolve the generic parameters.
                        this.visit_generics(generics);

                        // Resolve the items within the impl.
                        this.with_lifetime_rib(
                            LifetimeRibKind::AnonymousPassThrough(item_id),
                            |this| walk_list!(this, visit_assoc_item, impl_items, AssocCtxt::Impl),
                        );
                    },
                );
            },
        );
    }
}
