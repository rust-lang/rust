use rustc_ast::ptr::P;
use rustc_ast::visit::{self, Visitor};
use rustc_ast::{self as ast, attr, NodeId};
use rustc_ast_pretty::pprust;
use rustc_expand::base::{parse_macro_name_and_helper_attrs, ExtCtxt, ResolverExpand};
use rustc_expand::expand::{AstFragment, ExpansionConfig};
use rustc_session::Session;
use rustc_span::hygiene::AstPass;
use rustc_span::source_map::SourceMap;
use rustc_span::symbol::{kw, sym, Ident, Symbol};
use rustc_span::{Span, DUMMY_SP};
use smallvec::smallvec;
use std::mem;
use thin_vec::{thin_vec, ThinVec};

struct ProcMacroDerive {
    id: NodeId,
    trait_name: Symbol,
    function_name: Ident,
    span: Span,
    attrs: Vec<Symbol>,
}

struct ProcMacroDef {
    id: NodeId,
    function_name: Ident,
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
    handler: &'a rustc_errors::Handler,
    source_map: &'a SourceMap,
    is_proc_macro_crate: bool,
    is_test_crate: bool,
}

pub fn inject(
    krate: &mut ast::Crate,
    sess: &Session,
    resolver: &mut dyn ResolverExpand,
    is_proc_macro_crate: bool,
    has_proc_macro_decls: bool,
    is_test_crate: bool,
    handler: &rustc_errors::Handler,
) {
    let ecfg = ExpansionConfig::default("proc_macro".to_string());
    let mut cx = ExtCtxt::new(sess, ecfg, resolver, None);

    let mut collect = CollectProcMacros {
        macros: Vec::new(),
        in_root: true,
        handler,
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
            self.handler.span_err(
                sp,
                "`proc-macro` crate types currently cannot export any items other \
                    than functions tagged with `#[proc_macro]`, `#[proc_macro_derive]`, \
                    or `#[proc_macro_attribute]`",
            );
        }
    }

    fn collect_custom_derive(&mut self, item: &'a ast::Item, attr: &'a ast::Attribute) {
        let Some((trait_name, proc_attrs)) = parse_macro_name_and_helper_attrs(self.handler, attr, "derive") else {
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
            self.handler.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }

    fn collect_attr_proc_macro(&mut self, item: &'a ast::Item) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Attr(ProcMacroDef {
                id: item.id,
                span: item.span,
                function_name: item.ident,
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_attribute]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_attribute]` must be `pub`"
            };
            self.handler.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }

    fn collect_bang_proc_macro(&mut self, item: &'a ast::Item) {
        if self.in_root && item.vis.kind.is_pub() {
            self.macros.push(ProcMacro::Bang(ProcMacroDef {
                id: item.id,
                span: item.span,
                function_name: item.ident,
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro]` must be `pub`"
            };
            self.handler.span_err(self.source_map.guess_head_span(item.span), msg);
        }
    }
}

impl<'a> Visitor<'a> for CollectProcMacros<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        if let ast::ItemKind::MacroDef(..) = item.kind {
            if self.is_proc_macro_crate && attr::contains_name(&item.attrs, sym::macro_export) {
                let msg =
                    "cannot export macro_rules! macros from a `proc-macro` crate type currently";
                self.handler.span_err(self.source_map.guess_head_span(item.span), msg);
            }
        }

        // First up, make sure we're checking a bare function. If we're not then
        // we're just not interested in this item.
        //
        // If we find one, try to locate a `#[proc_macro_derive]` attribute on it.
        let is_fn = matches!(item.kind, ast::ItemKind::Fn(..));

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
                            "only one `#[{}]` attribute is allowed on any given function",
                            path_str,
                        )
                    } else {
                        format!(
                            "`#[{}]` and `#[{}]` attributes cannot both be applied
                            to the same function",
                            path_str,
                            pprust::path_to_string(&prev_item.path),
                        )
                    };

                    self.handler
                        .struct_span_err(attr.span, &msg)
                        .span_label(prev_attr.span, "previous attribute here")
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

        if !is_fn {
            let msg = format!(
                "the `#[{}]` attribute may only be used on bare functions",
                pprust::path_to_string(&attr.get_normal_item().path),
            );

            self.handler.span_err(attr.span, &msg);
            return;
        }

        if self.is_test_crate {
            return;
        }

        if !self.is_proc_macro_crate {
            let msg = format!(
                "the `#[{}]` attribute is only usable with crates of the `proc-macro` crate type",
                pprust::path_to_string(&attr.get_normal_item().path),
            );

            self.handler.span_err(attr.span, &msg);
            return;
        }

        if attr.has_name(sym::proc_macro_derive) {
            self.collect_custom_derive(item, attr);
        } else if attr.has_name(sym::proc_macro_attribute) {
            self.collect_attr_proc_macro(item);
        } else if attr.has_name(sym::proc_macro) {
            self.collect_bang_proc_macro(item);
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
                    cx.expr_call(
                        span,
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
                        ProcMacro::Derive(_) => unreachable!(),
                    };

                    cx.expr_call(
                        span,
                        proc_macro_ty_method_path(cx, ident),
                        thin_vec![
                            cx.expr_str(span, ca.function_name.name),
                            local_path(cx, ca.function_name),
                        ],
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
