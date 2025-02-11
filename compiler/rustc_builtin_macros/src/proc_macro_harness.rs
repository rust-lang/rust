use std::mem;

use rustc_ast::attr::{self, ProcMacroAttr};
use rustc_ast::mut_visit::{self, MutVisitor};
use rustc_ast::ptr::P;
use rustc_ast::{self as ast, NodeId};
use rustc_errors::DiagCtxtHandle;
use rustc_expand::base::{ExtCtxt, ResolverExpand, parse_macro_name_and_helper_attrs};
use rustc_expand::expand::{AstFragment, ExpansionConfig};
use rustc_feature::Features;
use rustc_session::Session;
use rustc_span::hygiene::AstPass;
use rustc_span::source_map::SourceMap;
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use smallvec::smallvec;
use thin_vec::{ThinVec, thin_vec};

use crate::errors;

struct ProcMacroDerive {
    id: NodeId,
    trait_name: Symbol,
    function_name: Ident,
    span: Span,
    attrs: Vec<Symbol>,
}

struct ProcMacroDef {
    id: NodeId,
    item_name: Ident,
    span: Span,
}

enum ProcMacro {
    Derive(ProcMacroDerive),
    Attr(ProcMacroDef),
    Bang(ProcMacroDef),
    Lint(ProcMacroDef),
}

struct CollectProcMacros<'a> {
    macros: Vec<ProcMacro>,
    in_root: bool,
    cx: ExtCtxt<'a>,
    dcx: DiagCtxtHandle<'a>,
    source_map: &'a SourceMap,
    is_proc_macro_crate: bool,
    is_test_crate: bool,
}

pub fn inject(
    krate: &mut ast::Crate,
    sess: &Session,
    features: &Features,
    resolver: &mut dyn ResolverExpand,
    is_proc_macro_crate: bool,
    has_proc_macro_decls: bool,
    is_test_crate: bool,
    dcx: DiagCtxtHandle<'_>,
) {
    let ecfg = ExpansionConfig::default("proc_macro".to_string(), features);
    let cx = ExtCtxt::new(sess, ecfg, resolver, None);

    let mut collect = CollectProcMacros {
        macros: Vec::new(),
        in_root: true,
        cx,
        dcx,
        source_map: sess.source_map(),
        is_proc_macro_crate,
        is_test_crate,
    };

    if has_proc_macro_decls || is_proc_macro_crate {
        mut_visit::walk_crate(&mut collect, krate);
    }

    if !is_proc_macro_crate {
        return;
    }

    if is_test_crate {
        return;
    }

    let decls = mk_decls(&mut collect.cx, &collect.macros);
    krate.items.push(decls);
}

impl CollectProcMacros<'_> {
    fn check_not_pub_in_root(&self, vis: &ast::Visibility, sp: Span) {
        if self.is_proc_macro_crate && self.in_root && vis.kind.is_pub() {
            self.dcx.emit_err(errors::ProcMacro { span: sp });
        }
    }

    fn collect_custom_derive(&mut self, item: &ast::Item, attr: &ast::Attribute) {
        let Some((trait_name, proc_attrs)) =
            parse_macro_name_and_helper_attrs(self.dcx, attr, "derive")
        else {
            return;
        };

        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Derive(ProcMacroDerive {
                id: item.id,
                span: item.span,
                trait_name,
                function_name: item.ident,
                attrs: proc_attrs,
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_derive]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_derive]` must be `pub`"
            };
            self.dcx.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }

    fn collect_attr_proc_macro(&mut self, item: &ast::Item) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Attr(ProcMacroDef {
                id: item.id,
                span: item.span,
                item_name: item.ident,
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_attribute]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_attribute]` must be `pub`"
            };
            self.dcx.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }

    fn collect_bang_proc_macro(&mut self, item: &ast::Item) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Bang(ProcMacroDef {
                id: item.id,
                span: item.span,
                item_name: item.ident,
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro]` must be `pub`"
            };
            self.dcx.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }

    fn collect_proc_macro_lint(&mut self, item: &ast::Item) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Lint(ProcMacroDef {
                id: item.id,
                span: item.span,
                item_name: item.ident,
            }));
        } else {
            let msg = if !self.in_root {
                "statics tagged with `#[proc_macro_lint]` must currently \
                 reside in the root of the crate"
            } else {
                "statics tagged with `#[proc_macro_lint]` must be `pub`"
            };
            self.dcx.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }
}

impl MutVisitor for CollectProcMacros<'_> {
    fn visit_item(&mut self, item: &mut P<ast::Item>) {
        let item_inner = &mut **item;

        if let ast::ItemKind::MacroDef(..) = item_inner.kind {
            if self.is_proc_macro_crate && attr::contains_name(&item_inner.attrs, sym::macro_export)
            {
                self.dcx.emit_err(errors::ExportMacroRules {
                    span: self.source_map.guess_head_span(item_inner.span),
                });
            }
        }

        let mut found_attr = None::<(&ast::Attribute, ProcMacroAttr)>;

        for attr in &item_inner.attrs {
            if let Some(kind) = attr.proc_macro_attr() {
                if let Some((prev_attr, prev_kind)) = found_attr {
                    let msg = if prev_kind == kind {
                        format!(
                            "only one `#[{kind}]` attribute is allowed on any given {descr}",
                            descr = item_inner.kind.descr(),
                        )
                    } else {
                        format!(
                            "`#[{kind}]` and `#[{prev_kind}]` attributes cannot both be applied \
                             to the same {descr}",
                            descr = item_inner.kind.descr(),
                        )
                    };

                    self.dcx
                        .struct_span_err(attr.span, msg)
                        .with_span_label(prev_attr.span, "previous attribute here")
                        .emit();

                    return;
                }

                found_attr = Some((attr, kind));
            }
        }

        let Some((attr, kind)) = found_attr else {
            self.check_not_pub_in_root(
                &item_inner.vis,
                self.source_map.guess_head_span(item_inner.span),
            );
            let prev_in_root = mem::replace(&mut self.in_root, false);
            mut_visit::walk_item(self, item);
            self.in_root = prev_in_root;
            return;
        };

        let mut extra_attrs = Vec::new();

        match kind {
            ProcMacroAttr::Derive | ProcMacroAttr::Attribute | ProcMacroAttr::Bang => {
                if !matches!(item_inner.kind, ast::ItemKind::Fn(..)) {
                    self.dcx
                        .create_err(errors::AttributeOnlyBeUsedOnBareFunctions {
                            span: attr.span,
                            path: &kind.to_string(),
                        })
                        .emit();
                    return;
                }
            }
            ProcMacroAttr::Lint => {
                let ast::ItemKind::Static(static_item) = &mut item_inner.kind else {
                    self.dcx
                        .create_err(errors::AttributeOnlyBeUsedOnStatics {
                            span: attr.span,
                            path: &kind.to_string(),
                        })
                        .emit();
                    return;
                };
                if let Some(expr) = &static_item.expr {
                    self.dcx
                        .create_err(errors::LintIdIsFilledInByAttribute {
                            span: static_item.ty.span.between(expr.span).to(expr.span),
                        })
                        .emit();
                } else {
                    // Expand `pub static lintname: LintId;` into:
                    //
                    //     #[allow(non_upper_case_globals)]
                    //     pub static lintname: LintId = ::proc_macro::LintId::new("lintname");
                    let expn_id = self.cx.resolver.expansion_for_ast_pass(
                        attr.span,
                        AstPass::ProcMacroHarness,
                        // Edition >2015 to be able to use ::proc_macro without generating an `extern crate`.
                        Some(rustc_span::edition::Edition::Edition2021),
                        &[sym::proc_macro_internals],
                        None,
                    );
                    let span = attr.span.with_def_site_ctxt(expn_id.to_expn_id());
                    extra_attrs.push(attr::mk_attr_nested_word(
                        &self.cx.psess().attr_id_generator,
                        ast::AttrStyle::Outer,
                        ast::Safety::Default,
                        sym::allow,
                        sym::non_upper_case_globals,
                        span,
                    ));
                    let lintid_new = self.cx.expr_call(
                        span,
                        self.cx.expr_path(self.cx.path(
                            span,
                            vec![
                                Ident::new(kw::PathRoot, span),
                                Ident::new(sym::proc_macro, span),
                                Ident::new(sym::LintId, span),
                                Ident::new(sym::new, span),
                            ],
                        )),
                        thin_vec![self.cx.expr_str(span, item_inner.ident.name)],
                    );
                    static_item.expr = Some(
                        self.cx
                            .monotonic_expander()
                            .fully_expand_fragment(AstFragment::Expr(lintid_new))
                            .make_expr(),
                    );
                }
            }
        }

        if self.is_test_crate {
            return;
        }

        if !self.is_proc_macro_crate {
            self.dcx
                .create_err(errors::AttributeOnlyUsableWithCrateType {
                    span: attr.span,
                    path: &kind.to_string(),
                })
                .emit();
            return;
        }

        match kind {
            ProcMacroAttr::Derive => self.collect_custom_derive(item_inner, attr),
            ProcMacroAttr::Attribute => self.collect_attr_proc_macro(item_inner),
            ProcMacroAttr::Bang => self.collect_bang_proc_macro(item_inner),
            ProcMacroAttr::Lint => self.collect_proc_macro_lint(item_inner),
        }

        // Insert #[allow(non_upper_case_globals)].
        // This needs to happen after `attr` is finished being borrowed above.
        item_inner.attrs.extend(extra_attrs);

        let prev_in_root = mem::replace(&mut self.in_root, false);
        mut_visit::walk_item(self, item);
        self.in_root = prev_in_root;
    }
}

// Creates a new module which looks like:
//
//      const _: () = {
//          extern crate proc_macro;
//
//          use proc_macro::bridge::client::ProcMacro;
//
//          #[rustc_proc_macro_decls]
//          #[used]
//          #[allow(deprecated)]
//          static DECLS: &[ProcMacro] = &[
//              ProcMacro::custom_derive($name_trait1, &[], ::$name1);
//              ProcMacro::custom_derive($name_trait2, &["attribute_name"], ::$name2);
//              // ...
//          ];
//      }
fn mk_decls(cx: &mut ExtCtxt<'_>, macros: &[ProcMacro]) -> P<ast::Item> {
    let expn_id = cx.resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::ProcMacroHarness,
        None,
        &[sym::rustc_attrs, sym::proc_macro_internals],
        None,
    );
    let span = DUMMY_SP.with_def_site_ctxt(expn_id.to_expn_id());

    let proc_macro = Ident::new(sym::proc_macro, span);
    let krate = cx.item(span, proc_macro, ast::AttrVec::new(), ast::ItemKind::ExternCrate(None));

    let bridge = Ident::new(sym::bridge, span);
    let client = Ident::new(sym::client, span);
    let proc_macro_ty = Ident::new(sym::ProcMacro, span);
    let custom_derive = Ident::new(sym::custom_derive, span);
    let attr = Ident::new(sym::attr, span);
    let bang = Ident::new(sym::bang, span);

    // We add NodeIds to 'resolver.proc_macros' in the order
    // that we generate expressions. The position of each NodeId
    // in the 'proc_macros' Vec corresponds to its position
    // in the static array that will be generated
    let decls = macros
        .iter()
        .map(|m| {
            let harness_span = span;
            let span = match m {
                ProcMacro::Derive(m) => m.span,
                ProcMacro::Attr(m) | ProcMacro::Bang(m) => m.span,
                ProcMacro::Lint(m) => m.span,
            };
            let local_path = |cx: &ExtCtxt<'_>, name| cx.expr_path(cx.path(span, vec![name]));
            let proc_macro_ty_method_path = |cx: &ExtCtxt<'_>, method| {
                cx.expr_path(cx.path(
                    span.with_ctxt(harness_span.ctxt()),
                    vec![proc_macro, bridge, client, proc_macro_ty, method],
                ))
            };
            match m {
                ProcMacro::Derive(cd) => {
                    cx.resolver.declare_proc_macro(cd.id);
                    // The call needs to use `harness_span` so that the const stability checker
                    // accepts it.
                    cx.expr_call(
                        harness_span,
                        proc_macro_ty_method_path(cx, custom_derive),
                        thin_vec![
                            cx.expr_str(span, cd.trait_name),
                            cx.expr_array_ref(
                                span,
                                cd.attrs
                                    .iter()
                                    .map(|&s| cx.expr_str(span, s))
                                    .collect::<ThinVec<_>>(),
                            ),
                            local_path(cx, cd.function_name),
                        ],
                    )
                }
                ProcMacro::Attr(ca) | ProcMacro::Bang(ca) => {
                    cx.resolver.declare_proc_macro(ca.id);
                    let ident = match m {
                        ProcMacro::Attr(_) => attr,
                        ProcMacro::Bang(_) => bang,
                        ProcMacro::Derive(_) | ProcMacro::Lint(_) => unreachable!(),
                    };

                    // The call needs to use `harness_span` so that the const stability checker
                    // accepts it.
                    cx.expr_call(
                        harness_span,
                        proc_macro_ty_method_path(cx, ident),
                        thin_vec![
                            cx.expr_str(span, ca.item_name.name),
                            local_path(cx, ca.item_name),
                        ],
                    )
                }
                ProcMacro::Lint(lint) => {
                    cx.resolver.declare_proc_macro(lint.id);
                    cx.expr_call(
                        harness_span,
                        proc_macro_ty_method_path(cx, Ident::new(sym::lint, span)),
                        thin_vec![local_path(cx, lint.item_name)],
                    )
                }
            }
        })
        .collect();

    let decls_static = cx
        .item_static(
            span,
            Ident::new(sym::_DECLS, span),
            cx.ty_ref(
                span,
                cx.ty(
                    span,
                    ast::TyKind::Slice(
                        cx.ty_path(cx.path(span, vec![proc_macro, bridge, client, proc_macro_ty])),
                    ),
                ),
                None,
                ast::Mutability::Not,
            ),
            ast::Mutability::Not,
            cx.expr_array_ref(span, decls),
        )
        .map(|mut i| {
            i.attrs.push(cx.attr_word(sym::rustc_proc_macro_decls, span));
            i.attrs.push(cx.attr_word(sym::used, span));
            i.attrs.push(cx.attr_nested_word(sym::allow, sym::deprecated, span));
            i
        });

    let block = cx.expr_block(
        cx.block(span, thin_vec![cx.stmt_item(span, krate), cx.stmt_item(span, decls_static)]),
    );

    let anon_constant = cx.item_const(
        span,
        Ident::new(kw::Underscore, span),
        cx.ty(span, ast::TyKind::Tup(ThinVec::new())),
        block,
    );

    // Integrate the new item into existing module structures.
    let items = AstFragment::Items(smallvec![anon_constant]);
    cx.monotonic_expander().fully_expand_fragment(items).make_items().pop().unwrap()
}
