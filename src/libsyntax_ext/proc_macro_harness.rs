use std::mem;

use smallvec::smallvec;
use syntax::ast::{self, Ident};
use syntax::attr;
use syntax::ext::base::ExtCtxt;
use syntax::ext::expand::{AstFragment, ExpansionConfig};
use syntax::ext::proc_macro::is_proc_macro_attr;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::sess::ParseSess;
use syntax::symbol::{kw, sym};
use syntax::visit::{self, Visitor};
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::hygiene::AstPass;

struct ProcMacroDerive {
    trait_name: ast::Name,
    function_name: Ident,
    span: Span,
    attrs: Vec<ast::Name>,
}

enum ProcMacroDefType {
    Attr,
    Bang
}

struct ProcMacroDef {
    function_name: Ident,
    span: Span,
    def_type: ProcMacroDefType
}

enum ProcMacro {
    Derive(ProcMacroDerive),
    Def(ProcMacroDef)
}

struct CollectProcMacros<'a> {
    macros: Vec<ProcMacro>,
    in_root: bool,
    handler: &'a errors::Handler,
    is_proc_macro_crate: bool,
    is_test_crate: bool,
}

pub fn inject(sess: &ParseSess,
              resolver: &mut dyn (::syntax::ext::base::Resolver),
              mut krate: ast::Crate,
              is_proc_macro_crate: bool,
              has_proc_macro_decls: bool,
              is_test_crate: bool,
              num_crate_types: usize,
              handler: &errors::Handler) -> ast::Crate {
    let ecfg = ExpansionConfig::default("proc_macro".to_string());
    let mut cx = ExtCtxt::new(sess, ecfg, resolver);

    let mut collect = CollectProcMacros {
        macros: Vec::new(),
        in_root: true,
        handler,
        is_proc_macro_crate,
        is_test_crate,
    };

    if has_proc_macro_decls || is_proc_macro_crate {
        visit::walk_crate(&mut collect, &krate);
    }
    // NOTE: If you change the order of macros in this vec
    // for any reason, you must also update 'raw_proc_macro'
    // in src/librustc_metadata/decoder.rs
    let macros = collect.macros;

    if !is_proc_macro_crate {
        return krate
    }

    if num_crate_types > 1 {
        handler.err("cannot mix `proc-macro` crate type with others");
    }

    if is_test_crate {
        return krate;
    }

    krate.module.items.push(mk_decls(&mut cx, &macros));

    krate
}

impl<'a> CollectProcMacros<'a> {
    fn check_not_pub_in_root(&self, vis: &ast::Visibility, sp: Span) {
        if self.is_proc_macro_crate && self.in_root && vis.node.is_pub() {
            self.handler.span_err(sp,
                                  "`proc-macro` crate types cannot \
                                   export any items other than functions \
                                   tagged with `#[proc_macro_derive]` currently");
        }
    }

    fn collect_custom_derive(&mut self, item: &'a ast::Item, attr: &'a ast::Attribute) {
        // Once we've located the `#[proc_macro_derive]` attribute, verify
        // that it's of the form `#[proc_macro_derive(Foo)]` or
        // `#[proc_macro_derive(Foo, attributes(A, ..))]`
        let list = match attr.meta_item_list() {
            Some(list) => list,
            None => return,
        };
        if list.len() != 1 && list.len() != 2 {
            self.handler.span_err(attr.span,
                                  "attribute must have either one or two arguments");
            return
        }
        let trait_attr = match list[0].meta_item() {
            Some(meta_item) => meta_item,
            _ => {
                self.handler.span_err(list[0].span(), "not a meta item");
                return
            }
        };
        let trait_ident = match trait_attr.ident() {
            Some(trait_ident) if trait_attr.is_word() => trait_ident,
            _ => {
                self.handler.span_err(trait_attr.span, "must only be one word");
                return
            }
        };

        if !trait_ident.name.can_be_raw() {
            self.handler.span_err(trait_attr.span,
                                  &format!("`{}` cannot be a name of derive macro", trait_ident));
        }

        let attributes_attr = list.get(1);
        let proc_attrs: Vec<_> = if let Some(attr) = attributes_attr {
            if !attr.check_name(sym::attributes) {
                self.handler.span_err(attr.span(), "second argument must be `attributes`")
            }
            attr.meta_item_list().unwrap_or_else(|| {
                self.handler.span_err(attr.span(),
                                      "attribute must be of form: `attributes(foo, bar)`");
                &[]
            }).into_iter().filter_map(|attr| {
                let attr = match attr.meta_item() {
                    Some(meta_item) => meta_item,
                    _ => {
                        self.handler.span_err(attr.span(), "not a meta item");
                        return None;
                    }
                };

                let ident = match attr.ident() {
                    Some(ident) if attr.is_word() => ident,
                    _ => {
                        self.handler.span_err(attr.span, "must only be one word");
                        return None;
                    }
                };
                if !ident.name.can_be_raw() {
                    self.handler.span_err(
                        attr.span,
                        &format!("`{}` cannot be a name of derive helper attribute", ident),
                    );
                }

                Some(ident.name)
            }).collect()
        } else {
            Vec::new()
        };

        if self.in_root && item.vis.node.is_pub() {
            self.macros.push(ProcMacro::Derive(ProcMacroDerive {
                span: item.span,
                trait_name: trait_ident.name,
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
            self.handler.span_err(item.span, msg);
        }
    }

    fn collect_attr_proc_macro(&mut self, item: &'a ast::Item) {
        if self.in_root && item.vis.node.is_pub() {
            self.macros.push(ProcMacro::Def(ProcMacroDef {
                span: item.span,
                function_name: item.ident,
                def_type: ProcMacroDefType::Attr
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro_attribute]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro_attribute]` must be `pub`"
            };
            self.handler.span_err(item.span, msg);
        }
    }

    fn collect_bang_proc_macro(&mut self, item: &'a ast::Item) {
        if self.in_root && item.vis.node.is_pub() {
            self.macros.push(ProcMacro::Def(ProcMacroDef {
                span: item.span,
                function_name: item.ident,
                def_type: ProcMacroDefType::Bang
            }));
        } else {
            let msg = if !self.in_root {
                "functions tagged with `#[proc_macro]` must \
                 currently reside in the root of the crate"
            } else {
                "functions tagged with `#[proc_macro]` must be `pub`"
            };
            self.handler.span_err(item.span, msg);
        }
    }
}

impl<'a> Visitor<'a> for CollectProcMacros<'a> {
    fn visit_item(&mut self, item: &'a ast::Item) {
        if let ast::ItemKind::MacroDef(..) = item.kind {
            if self.is_proc_macro_crate && attr::contains_name(&item.attrs, sym::macro_export) {
                let msg =
                    "cannot export macro_rules! macros from a `proc-macro` crate type currently";
                self.handler.span_err(item.span, msg);
            }
        }

        // First up, make sure we're checking a bare function. If we're not then
        // we're just not interested in this item.
        //
        // If we find one, try to locate a `#[proc_macro_derive]` attribute on it.
        let is_fn = match item.kind {
            ast::ItemKind::Fn(..) => true,
            _ => false,
        };

        let mut found_attr: Option<&'a ast::Attribute> = None;

        for attr in &item.attrs {
            if is_proc_macro_attr(&attr) {
                if let Some(prev_attr) = found_attr {
                    let path_str = pprust::path_to_string(&attr.path);
                    let msg = if attr.path.segments[0].ident.name ==
                                 prev_attr.path.segments[0].ident.name {
                        format!(
                            "only one `#[{}]` attribute is allowed on any given function",
                            path_str,
                        )
                    } else {
                        format!(
                            "`#[{}]` and `#[{}]` attributes cannot both be applied
                            to the same function",
                            path_str,
                            pprust::path_to_string(&prev_attr.path),
                        )
                    };

                    self.handler.struct_span_err(attr.span, &msg)
                        .span_note(prev_attr.span, "previous attribute here")
                        .emit();

                    return;
                }

                found_attr = Some(attr);
            }
        }

        let attr = match found_attr {
            None => {
                self.check_not_pub_in_root(&item.vis, item.span);
                let prev_in_root = mem::replace(&mut self.in_root, false);
                visit::walk_item(self, item);
                self.in_root = prev_in_root;
                return;
            },
            Some(attr) => attr,
        };

        if !is_fn {
            let msg = format!(
                "the `#[{}]` attribute may only be used on bare functions",
                pprust::path_to_string(&attr.path),
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
                pprust::path_to_string(&attr.path),
            );

            self.handler.span_err(attr.span, &msg);
            return;
        }

        if attr.check_name(sym::proc_macro_derive) {
            self.collect_custom_derive(item, attr);
        } else if attr.check_name(sym::proc_macro_attribute) {
            self.collect_attr_proc_macro(item);
        } else if attr.check_name(sym::proc_macro) {
            self.collect_bang_proc_macro(item);
        };

        let prev_in_root = mem::replace(&mut self.in_root, false);
        visit::walk_item(self, item);
        self.in_root = prev_in_root;
    }

    fn visit_mac(&mut self, mac: &'a ast::Mac) {
        visit::walk_mac(self, mac)
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
//          static DECLS: &[ProcMacro] = &[
//              ProcMacro::custom_derive($name_trait1, &[], ::$name1);
//              ProcMacro::custom_derive($name_trait2, &["attribute_name"], ::$name2);
//              // ...
//          ];
//      }
fn mk_decls(
    cx: &mut ExtCtxt<'_>,
    macros: &[ProcMacro],
) -> P<ast::Item> {
    let expn_id = cx.resolver.expansion_for_ast_pass(
        DUMMY_SP,
        AstPass::ProcMacroHarness,
        &[sym::rustc_attrs, sym::proc_macro_internals],
        None,
    );
    let span = DUMMY_SP.with_def_site_ctxt(expn_id);

    let proc_macro = Ident::new(sym::proc_macro, span);
    let krate = cx.item(span,
                        proc_macro,
                        Vec::new(),
                        ast::ItemKind::ExternCrate(None));

    let bridge = cx.ident_of("bridge", span);
    let client = cx.ident_of("client", span);
    let proc_macro_ty = cx.ident_of("ProcMacro", span);
    let custom_derive = cx.ident_of("custom_derive", span);
    let attr = cx.ident_of("attr", span);
    let bang = cx.ident_of("bang", span);

    let decls = {
        let local_path = |sp: Span, name| {
            cx.expr_path(cx.path(sp.with_ctxt(span.ctxt()), vec![name]))
        };
        let proc_macro_ty_method_path = |method| cx.expr_path(cx.path(span, vec![
            proc_macro, bridge, client, proc_macro_ty, method,
        ]));
        macros.iter().map(|m| {
            match m {
                ProcMacro::Derive(cd) => {
                    cx.expr_call(span, proc_macro_ty_method_path(custom_derive), vec![
                        cx.expr_str(cd.span, cd.trait_name),
                        cx.expr_vec_slice(
                            span,
                            cd.attrs.iter().map(|&s| cx.expr_str(cd.span, s)).collect::<Vec<_>>()
                        ),
                        local_path(cd.span, cd.function_name),
                    ])
                },
                ProcMacro::Def(ca) => {
                    let ident = match ca.def_type {
                        ProcMacroDefType::Attr => attr,
                        ProcMacroDefType::Bang => bang
                    };

                    cx.expr_call(span, proc_macro_ty_method_path(ident), vec![
                        cx.expr_str(ca.span, ca.function_name.name),
                        local_path(ca.span, ca.function_name),
                    ])

                }
            }
        }).collect()
    };

    let decls_static = cx.item_static(
        span,
        cx.ident_of("_DECLS", span),
        cx.ty_rptr(span,
            cx.ty(span, ast::TyKind::Slice(
                cx.ty_path(cx.path(span,
                    vec![proc_macro, bridge, client, proc_macro_ty])))),
            None, ast::Mutability::Immutable),
        ast::Mutability::Immutable,
        cx.expr_vec_slice(span, decls),
    ).map(|mut i| {
        let attr = cx.meta_word(span, sym::rustc_proc_macro_decls);
        i.attrs.push(cx.attribute(attr));
        i
    });

    let block = cx.expr_block(cx.block(
        span,
        vec![cx.stmt_item(span, krate), cx.stmt_item(span, decls_static)],
    ));

    let anon_constant = cx.item_const(
        span,
        ast::Ident::new(kw::Underscore, span),
        cx.ty(span, ast::TyKind::Tup(Vec::new())),
        block,
    );

    // Integrate the new item into existing module structures.
    let items = AstFragment::Items(smallvec![anon_constant]);
    cx.monotonic_expander().fully_expand_fragment(items).make_items().pop().unwrap()
}
