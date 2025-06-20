use std::mem;

use rustc_ast::ptr::P;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, NodeId, attr};
use rustc_ast_pretty::pprust;
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
    function_ident: Ident,
    span: Span,
    attrs: Vec<Symbol>,
}

struct ProcMacroDef {
    id: NodeId,
    function_ident: Ident,
    span: Span,
}

enum ProcMacro {
    Derive(ProcMacroDerive),
    Attr(ProcMacroDef),
    Bang(ProcMacroDef),
}

struct CollectProcMacros<'a> {
    macros: Vec<ProcMacro>,
    in_root: bool,
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
    let ecfg = ExpansionConfig::default(sym::proc_macro, features);
    let mut cx = ExtCtxt::new(sess, ecfg, resolver, None);

    let mut collect = CollectProcMacros {
        macros: Vec::new(),
        in_root: true,
        dcx,
        source_map: sess.source_map(),
        is_proc_macro_crate,
        is_test_crate,
    };

    if has_proc_macro_decls || is_proc_macro_crate {
        visit::walk_crate(&mut collect, krate);
    }
    let macros = collect.macros;

    if !is_proc_macro_crate {
        return;
    }

    if is_test_crate {
        return;
    }

    let decls = mk_decls(&mut cx, &macros);
    krate.items.push(decls);
}

impl<'a> CollectProcMacros<'a> {
    fn check_not_pub_in_root(&self, vis: &ast::Visibility, sp: Span) {
        if self.is_proc_macro_crate && self.in_root && vis.kind.is_pub() {
            self.dcx.emit_err(errors::ProcMacro { span: sp });
        }
    }

    fn collect_custom_derive(
        &mut self,
        item: &'a ast::Item,
        function_ident: Ident,
        attr: &'a ast::Attribute,
    ) {
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
                function_ident,
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

    fn collect_attr_proc_macro(&mut self, item: &'a ast::Item, function_ident: Ident) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Attr(ProcMacroDef {
                id: item.id,
                span: item.span,
                function_ident,
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

    fn collect_bang_proc_macro(&mut self, item: &'a ast::Item, function_ident: Ident) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Bang(ProcMacroDef {
                id: item.id,
                span: item.span,
                function_ident,
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
}

impl<'a> Visitor<'a> for CollectProcMacros<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        if let ast::ItemKind::MacroDef(..) = item.kind {
            if self.is_proc_macro_crate && attr::contains_name(&item.attrs, sym::macro_export) {
                self.dcx.emit_err(errors::ExportMacroRules {
                    span: self.source_map.guess_head_span(item.span),
                });
            }
        }

        let mut found_attr: Option<&'a ast::Attribute> = None;

        for attr in &item.attrs {
            if attr.is_proc_macro_attr() {
                if let Some(prev_attr) = found_attr {
                    let prev_item = prev_attr.get_normal_item();
                    let item = attr.get_normal_item();
                    let path_str = pprust::path_to_string(&item.path);
                    let msg = if item.path.segments[0].ident.name
                        == prev_item.path.segments[0].ident.name
                    {
                        format!(
                            "only one `#[{path_str}]` attribute is allowed on any given function",
                        )
                    } else {
                        format!(
                            "`#[{}]` and `#[{}]` attributes cannot both be applied
                            to the same function",
                            path_str,
                            pprust::path_to_string(&prev_item.path),
                        )
                    };

                    self.dcx
                        .struct_span_err(attr.span, msg)
                        .with_span_label(prev_attr.span, "previous attribute here")
                        .emit();

                    return;
                }

                found_attr = Some(attr);
            }
        }

        let Some(attr) = found_attr else {
            self.check_not_pub_in_root(&item.vis, self.source_map.guess_head_span(item.span));
            let prev_in_root = mem::replace(&mut self.in_root, false);
            visit::walk_item(self, item);
            self.in_root = prev_in_root;
            return;
        };

        // Make sure we're checking a bare function. If we're not then we're
        // just not interested any further in this item.
        let fn_ident = if let ast::ItemKind::Fn(fn_) = &item.kind {
            fn_.ident
        } else {
            self.dcx
                .create_err(errors::AttributeOnlyBeUsedOnBareFunctions {
                    span: attr.span,
                    path: &pprust::path_to_string(&attr.get_normal_item().path),
                })
                .emit();
            return;
        };

        if self.is_test_crate {
            return;
        }

        if !self.is_proc_macro_crate {
            self.dcx
                .create_err(errors::AttributeOnlyUsableWithCrateType {
                    span: attr.span,
                    path: &pprust::path_to_string(&attr.get_normal_item().path),
                })
                .emit();
            return;
        }

        // Try to locate a `#[proc_macro_derive]` attribute.
        if attr.has_name(sym::proc_macro_derive) {
            self.collect_custom_derive(item, fn_ident, attr);
        } else if attr.has_name(sym::proc_macro_attribute) {
            self.collect_attr_proc_macro(item, fn_ident);
        } else if attr.has_name(sym::proc_macro) {
            self.collect_bang_proc_macro(item, fn_ident);
        };

        let prev_in_root = mem::replace(&mut self.in_root, false);
        visit::walk_item(self, item);
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
        &[sym::rustc_attrs, sym::proc_macro_internals],
        None,
    );
    let span = DUMMY_SP.with_def_site_ctxt(expn_id.to_expn_id());

    let proc_macro = Ident::new(sym::proc_macro, span);
    let krate = cx.item(span, ast::AttrVec::new(), ast::ItemKind::ExternCrate(None, proc_macro));

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
            };
            let local_path = |cx: &ExtCtxt<'_>, ident| cx.expr_path(cx.path(span, vec![ident]));
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
                            local_path(cx, cd.function_ident),
                        ],
                    )
                }
                ProcMacro::Attr(ca) | ProcMacro::Bang(ca) => {
                    cx.resolver.declare_proc_macro(ca.id);
                    let ident = match m {
                        ProcMacro::Attr(_) => attr,
                        ProcMacro::Bang(_) => bang,
                        ProcMacro::Derive(_) => unreachable!(),
                    };

                    // The call needs to use `harness_span` so that the const stability checker
                    // accepts it.
                    cx.expr_call(
                        harness_span,
                        proc_macro_ty_method_path(cx, ident),
                        thin_vec![
                            cx.expr_str(span, ca.function_ident.name),
                            local_path(cx, ca.function_ident),
                        ],
                    )
                }
            }
        })
        .collect();

    let mut decls_static = cx.item_static(
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
    );
    decls_static.attrs.extend([
        cx.attr_word(sym::rustc_proc_macro_decls, span),
        cx.attr_word(sym::used, span),
        cx.attr_nested_word(sym::allow, sym::deprecated, span),
    ]);

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
