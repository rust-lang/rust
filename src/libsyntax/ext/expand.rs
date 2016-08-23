// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{Block, Crate, Ident, Mac_, PatKind};
use ast::{MacStmtStyle, Stmt, StmtKind, ItemKind};
use ast;
use ext::hygiene::Mark;
use attr::{self, HasAttrs};
use codemap::{dummy_spanned, ExpnInfo, NameAndSpan, MacroBang, MacroAttribute};
use syntax_pos::{self, Span, ExpnId};
use config::StripUnconfigured;
use ext::base::*;
use feature_gate::{self, Features};
use fold;
use fold::*;
use parse::token::{intern, keywords};
use ptr::P;
use tokenstream::TokenTree;
use util::small_vector::SmallVector;
use visit;
use visit::Visitor;
use std_inject;

// A trait for AST nodes and AST node lists into which macro invocations may expand.
trait MacroGenerable: Sized {
    // Expand the given MacResult using its appropriate `make_*` method.
    fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self>;

    // Fold this node or list of nodes using the given folder.
    fn fold_with<F: Folder>(self, folder: &mut F) -> Self;
    fn visit_with<V: Visitor>(&self, visitor: &mut V);

    // The user-friendly name of the node type (e.g. "expression", "item", etc.) for diagnostics.
    fn kind_name() -> &'static str;

    // Return a placeholder expansion to allow compilation to continue after an erroring expansion.
    fn dummy(span: Span) -> Self {
        Self::make_with(DummyResult::any(span)).unwrap()
    }
}

macro_rules! impl_macro_generable {
    ($($ty:ty: $kind_name:expr, .$make:ident,
               $(.$fold:ident)*  $(lift .$fold_elt:ident)*,
               $(.$visit:ident)* $(lift .$visit_elt:ident)*;)*) => { $(
        impl MacroGenerable for $ty {
            fn kind_name() -> &'static str { $kind_name }
            fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self> { result.$make() }
            fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
                $( folder.$fold(self) )*
                $( self.into_iter().flat_map(|item| folder. $fold_elt (item)).collect() )*
            }
            fn visit_with<V: Visitor>(&self, visitor: &mut V) {
                $( visitor.$visit(self) )*
                $( for item in self.as_slice() { visitor. $visit_elt (item) } )*
            }
        }
    )* }
}

impl_macro_generable! {
    P<ast::Expr>: "expression", .make_expr, .fold_expr, .visit_expr;
    P<ast::Pat>:  "pattern",    .make_pat,  .fold_pat,  .visit_pat;
    P<ast::Ty>:   "type",       .make_ty,   .fold_ty,   .visit_ty;
    SmallVector<ast::Stmt>: "statement", .make_stmts, lift .fold_stmt, lift .visit_stmt;
    SmallVector<P<ast::Item>>: "item",   .make_items, lift .fold_item, lift .visit_item;
    SmallVector<ast::TraitItem>:
        "trait item", .make_trait_items, lift .fold_trait_item, lift .visit_trait_item;
    SmallVector<ast::ImplItem>:
        "impl item",  .make_impl_items,  lift .fold_impl_item,  lift .visit_impl_item;
}

impl MacroGenerable for Option<P<ast::Expr>> {
    fn kind_name() -> &'static str { "expression" }
    fn make_with<'a>(result: Box<MacResult + 'a>) -> Option<Self> {
        result.make_expr().map(Some)
    }
    fn fold_with<F: Folder>(self, folder: &mut F) -> Self {
        self.and_then(|expr| folder.fold_opt_expr(expr))
    }
    fn visit_with<V: Visitor>(&self, visitor: &mut V) {
        self.as_ref().map(|expr| visitor.visit_expr(expr));
    }
}

pub fn expand_expr(expr: ast::Expr, fld: &mut MacroExpander) -> P<ast::Expr> {
    match expr.node {
        // expr_mac should really be expr_ext or something; it's the
        // entry-point for all syntax extensions.
        ast::ExprKind::Mac(mac) => {
            return expand_mac_invoc(mac, None, expr.attrs.into(), expr.span, fld);
        }
        _ => P(noop_fold_expr(expr, fld)),
    }
}

struct MacroScopePlaceholder;
impl MacResult for MacroScopePlaceholder {
    fn make_items(self: Box<Self>) -> Option<SmallVector<P<ast::Item>>> {
        Some(SmallVector::one(P(ast::Item {
            ident: keywords::Invalid.ident(),
            attrs: Vec::new(),
            id: ast::DUMMY_NODE_ID,
            node: ast::ItemKind::Mac(dummy_spanned(ast::Mac_ {
                path: ast::Path { span: syntax_pos::DUMMY_SP, global: false, segments: Vec::new() },
                tts: Vec::new(),
            })),
            vis: ast::Visibility::Inherited,
            span: syntax_pos::DUMMY_SP,
        })))
    }
}

/// Expand a macro invocation. Returns the result of expansion.
fn expand_mac_invoc<T>(mac: ast::Mac, ident: Option<Ident>, attrs: Vec<ast::Attribute>, span: Span,
                       fld: &mut MacroExpander) -> T
    where T: MacroGenerable,
{
    // It would almost certainly be cleaner to pass the whole macro invocation in,
    // rather than pulling it apart and marking the tts and the ctxt separately.
    let Mac_ { path, tts, .. } = mac.node;
    let mark = Mark::fresh();

    fn mac_result<'a>(path: &ast::Path, ident: Option<Ident>, tts: Vec<TokenTree>, mark: Mark,
                      attrs: Vec<ast::Attribute>, call_site: Span, fld: &'a mut MacroExpander)
                      -> Option<Box<MacResult + 'a>> {
        // Detect use of feature-gated or invalid attributes on macro invoations
        // since they will not be detected after macro expansion.
        for attr in attrs.iter() {
            feature_gate::check_attribute(&attr, &fld.cx.parse_sess.span_diagnostic,
                                          &fld.cx.parse_sess.codemap(),
                                          &fld.cx.ecfg.features.unwrap());
        }

        if path.segments.len() > 1 || path.global || !path.segments[0].parameters.is_empty() {
            fld.cx.span_err(path.span, "expected macro name without module separators");
            return None;
        }

        let extname = path.segments[0].identifier.name;
        let extension = if let Some(extension) = fld.cx.syntax_env.find(extname) {
            extension
        } else {
            let mut err = fld.cx.struct_span_err(path.span,
                                                 &format!("macro undefined: '{}!'", &extname));
            fld.cx.suggest_macro_name(&extname.as_str(), &mut err);
            err.emit();
            return None;
        };

        let ident = ident.unwrap_or(keywords::Invalid.ident());
        let marked_tts = mark_tts(&tts, mark);
        match *extension {
            NormalTT(ref expandfun, exp_span, allow_internal_unstable) => {
                if ident.name != keywords::Invalid.name() {
                    let msg =
                        format!("macro {}! expects no ident argument, given '{}'", extname, ident);
                    fld.cx.span_err(path.span, &msg);
                    return None;
                }

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: exp_span,
                        allow_internal_unstable: allow_internal_unstable,
                    },
                });

                Some(expandfun.expand(fld.cx, call_site, &marked_tts))
            }

            IdentTT(ref expander, tt_span, allow_internal_unstable) => {
                if ident.name == keywords::Invalid.name() {
                    fld.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return None;
                };

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: tt_span,
                        allow_internal_unstable: allow_internal_unstable,
                    }
                });

                Some(expander.expand(fld.cx, call_site, ident, marked_tts))
            }

            MacroRulesTT => {
                if ident.name == keywords::Invalid.name() {
                    fld.cx.span_err(path.span,
                                    &format!("macro {}! expects an ident argument", extname));
                    return None;
                };

                fld.cx.bt_push(ExpnInfo {
                    call_site: call_site,
                    callee: NameAndSpan {
                        format: MacroBang(extname),
                        span: None,
                        // `macro_rules!` doesn't directly allow unstable
                        // (this is orthogonal to whether the macro it creates allows it)
                        allow_internal_unstable: false,
                    }
                });

                let def = ast::MacroDef {
                    ident: ident,
                    id: ast::DUMMY_NODE_ID,
                    span: call_site,
                    imported_from: None,
                    use_locally: true,
                    body: marked_tts,
                    export: attr::contains_name(&attrs, "macro_export"),
                    allow_internal_unstable: attr::contains_name(&attrs, "allow_internal_unstable"),
                    attrs: attrs,
                };

                fld.cx.insert_macro(def.clone());

                // macro_rules! has a side effect, but expands to nothing.
                // If keep_macs is true, expands to a MacEager::items instead.
                if fld.keep_macs {
                    Some(MacEager::items(SmallVector::one(P(ast::Item {
                        ident: def.ident,
                        attrs: def.attrs.clone(),
                        id: ast::DUMMY_NODE_ID,
                        node: ast::ItemKind::Mac(ast::Mac {
                            span: def.span,
                            node: ast::Mac_ {
                                path: path.clone(),
                                tts: def.body.clone(),
                            }
                        }),
                        vis: ast::Visibility::Inherited,
                        span: def.span,
                    }))))
                } else {
                    Some(Box::new(MacroScopePlaceholder))
                }
            }

            MultiDecorator(..) | MultiModifier(..) => {
                fld.cx.span_err(path.span,
                                &format!("`{}` can only be used in attributes", extname));
                None
            }
        }
    }

    let opt_expanded = T::make_with(match mac_result(&path, ident, tts, mark, attrs, span, fld) {
        Some(result) => result,
        None => return T::dummy(span),
    });

    let expanded = if let Some(expanded) = opt_expanded {
        expanded
    } else {
        let msg = format!("non-{kind} macro in {kind} position: {name}",
                          name = path.segments[0].identifier.name, kind = T::kind_name());
        fld.cx.span_err(path.span, &msg);
        return T::dummy(span);
    };

    let marked = expanded.fold_with(&mut Marker { mark: mark, expn_id: Some(fld.cx.backtrace()) });
    let configured = marked.fold_with(&mut fld.strip_unconfigured());
    fld.load_macros(&configured);

    let fully_expanded = if fld.single_step {
        configured
    } else {
        configured.fold_with(fld)
    };

    fld.cx.bt_pop();
    fully_expanded
}

// eval $e with a new exts frame.
// must be a macro so that $e isn't evaluated too early.
macro_rules! with_exts_frame {
    ($extsboxexpr:expr,$macros_escape:expr,$e:expr) =>
    ({$extsboxexpr.push_frame();
      $extsboxexpr.info().macros_escape = $macros_escape;
      let result = $e;
      $extsboxexpr.pop_frame();
      result
     })
}

// When we enter a module, record it, for the sake of `module!`
pub fn expand_item(it: P<ast::Item>, fld: &mut MacroExpander)
                   -> SmallVector<P<ast::Item>> {
    expand_annotatable(Annotatable::Item(it), fld)
        .into_iter().map(|i| i.expect_item()).collect()
}

// does this attribute list contain "macro_use" ?
fn contains_macro_use(fld: &mut MacroExpander, attrs: &[ast::Attribute]) -> bool {
    for attr in attrs {
        let mut is_use = attr.check_name("macro_use");
        if attr.check_name("macro_escape") {
            let mut err =
                fld.cx.struct_span_warn(attr.span,
                                        "macro_escape is a deprecated synonym for macro_use");
            is_use = true;
            if let ast::AttrStyle::Inner = attr.node.style {
                err.help("consider an outer attribute, \
                          #[macro_use] mod ...").emit();
            } else {
                err.emit();
            }
        };

        if is_use {
            if !attr.is_word() {
              fld.cx.span_err(attr.span, "arguments to macro_use are not allowed here");
            }
            return true;
        }
    }
    false
}

/// Expand a stmt
fn expand_stmt(stmt: Stmt, fld: &mut MacroExpander) -> SmallVector<Stmt> {
    let (mac, style, attrs) = match stmt.node {
        StmtKind::Mac(mac) => mac.unwrap(),
        _ => return noop_fold_stmt(stmt, fld)
    };

    let mut fully_expanded: SmallVector<ast::Stmt> =
        expand_mac_invoc(mac, None, attrs.into(), stmt.span, fld);

    // If this is a macro invocation with a semicolon, then apply that
    // semicolon to the final statement produced by expansion.
    if style == MacStmtStyle::Semicolon {
        if let Some(stmt) = fully_expanded.pop() {
            fully_expanded.push(stmt.add_trailing_semicolon());
        }
    }

    fully_expanded
}

fn expand_pat(p: P<ast::Pat>, fld: &mut MacroExpander) -> P<ast::Pat> {
    match p.node {
        PatKind::Mac(_) => {}
        _ => return noop_fold_pat(p, fld)
    }
    p.and_then(|ast::Pat {node, span, ..}| {
        match node {
            PatKind::Mac(mac) => expand_mac_invoc(mac, None, Vec::new(), span, fld),
            _ => unreachable!()
        }
    })
}

fn expand_multi_modified(a: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    match a {
        Annotatable::Item(it) => match it.node {
            ast::ItemKind::Mac(..) => {
                if match it.node {
                    ItemKind::Mac(ref mac) => mac.node.path.segments.is_empty(),
                    _ => unreachable!(),
                } {
                    return SmallVector::one(Annotatable::Item(it));
                }
                it.and_then(|it| match it.node {
                    ItemKind::Mac(mac) =>
                        expand_mac_invoc(mac, Some(it.ident), it.attrs, it.span, fld),
                    _ => unreachable!(),
                })
            }
            ast::ItemKind::Mod(_) | ast::ItemKind::ForeignMod(_) => {
                let valid_ident =
                    it.ident.name != keywords::Invalid.name();

                if valid_ident {
                    fld.cx.mod_push(it.ident);
                }
                let macro_use = contains_macro_use(fld, &it.attrs);
                let result = with_exts_frame!(fld.cx.syntax_env,
                                              macro_use,
                                              noop_fold_item(it, fld));
                if valid_ident {
                    fld.cx.mod_pop();
                }
                result
            },
            _ => noop_fold_item(it, fld),
        }.into_iter().map(|i| Annotatable::Item(i)).collect(),

        Annotatable::TraitItem(it) => {
            expand_trait_item(it.unwrap(), fld).into_iter().
                map(|it| Annotatable::TraitItem(P(it))).collect()
        }

        Annotatable::ImplItem(ii) => {
            expand_impl_item(ii.unwrap(), fld).into_iter().
                map(|ii| Annotatable::ImplItem(P(ii))).collect()
        }
    }
}

fn expand_annotatable(mut item: Annotatable, fld: &mut MacroExpander) -> SmallVector<Annotatable> {
    let mut multi_modifier = None;
    item = item.map_attrs(|mut attrs| {
        for i in 0..attrs.len() {
            if let Some(extension) = fld.cx.syntax_env.find(intern(&attrs[i].name())) {
                match *extension {
                    MultiModifier(..) | MultiDecorator(..) => {
                        multi_modifier = Some((attrs.remove(i), extension));
                        break;
                    }
                    _ => {}
                }
            }
        }
        attrs
    });

    match multi_modifier {
        None => expand_multi_modified(item, fld),
        Some((attr, extension)) => {
            attr::mark_used(&attr);
            fld.cx.bt_push(ExpnInfo {
                call_site: attr.span,
                callee: NameAndSpan {
                    format: MacroAttribute(intern(&attr.name())),
                    span: Some(attr.span),
                    allow_internal_unstable: false,
                }
            });

            let modified = match *extension {
                MultiModifier(ref mac) => mac.expand(fld.cx, attr.span, &attr.node.value, item),
                MultiDecorator(ref mac) => {
                    let mut items = Vec::new();
                    mac.expand(fld.cx, attr.span, &attr.node.value, &item,
                               &mut |item| items.push(item));
                    items.push(item);
                    items
                }
                _ => unreachable!(),
            };

            fld.cx.bt_pop();
            let configured = modified.into_iter().flat_map(|it| {
                it.fold_with(&mut fld.strip_unconfigured())
            }).collect::<SmallVector<_>>();

            configured.into_iter().flat_map(|it| expand_annotatable(it, fld)).collect()
        }
    }
}

fn expand_impl_item(ii: ast::ImplItem, fld: &mut MacroExpander)
                 -> SmallVector<ast::ImplItem> {
    match ii.node {
        ast::ImplItemKind::Macro(mac) => {
            expand_mac_invoc(mac, None, ii.attrs, ii.span, fld)
        }
        _ => fold::noop_fold_impl_item(ii, fld)
    }
}

fn expand_trait_item(ti: ast::TraitItem, fld: &mut MacroExpander)
                     -> SmallVector<ast::TraitItem> {
    match ti.node {
        ast::TraitItemKind::Macro(mac) => {
            expand_mac_invoc(mac, None, ti.attrs, ti.span, fld)
        }
        _ => fold::noop_fold_trait_item(ti, fld)
    }
}

pub fn expand_type(t: P<ast::Ty>, fld: &mut MacroExpander) -> P<ast::Ty> {
    let t = match t.node.clone() {
        ast::TyKind::Mac(mac) => {
            expand_mac_invoc(mac, None, Vec::new(), t.span, fld)
        }
        _ => t
    };

    fold::noop_fold_ty(t, fld)
}

/// A tree-folder that performs macro expansion
pub struct MacroExpander<'a, 'b:'a> {
    pub cx: &'a mut ExtCtxt<'b>,
    pub single_step: bool,
    pub keep_macs: bool,
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    pub fn new(cx: &'a mut ExtCtxt<'b>,
               single_step: bool,
               keep_macs: bool) -> MacroExpander<'a, 'b> {
        MacroExpander {
            cx: cx,
            single_step: single_step,
            keep_macs: keep_macs
        }
    }

    fn strip_unconfigured(&mut self) -> StripUnconfigured {
        StripUnconfigured {
            config: &self.cx.cfg,
            should_test: self.cx.ecfg.should_test,
            sess: self.cx.parse_sess,
            features: self.cx.ecfg.features,
        }
    }

    fn load_macros<T: MacroGenerable>(&mut self, node: &T) {
        struct MacroLoadingVisitor<'a, 'b: 'a>{
            cx: &'a mut ExtCtxt<'b>,
            at_crate_root: bool,
        }

        impl<'a, 'b> Visitor for MacroLoadingVisitor<'a, 'b> {
            fn visit_mac(&mut self, _: &ast::Mac) {}
            fn visit_item(&mut self, item: &ast::Item) {
                if let ast::ItemKind::ExternCrate(..) = item.node {
                    // We need to error on `#[macro_use] extern crate` when it isn't at the
                    // crate root, because `$crate` won't work properly.
                    for def in self.cx.loader.load_crate(item, self.at_crate_root) {
                        match def {
                            LoadedMacro::Def(def) => self.cx.insert_macro(def),
                            LoadedMacro::CustomDerive(name, ext) => {
                                self.cx.insert_custom_derive(&name, ext, item.span);
                            }
                        }
                    }
                } else {
                    let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                    visit::walk_item(self, item);
                    self.at_crate_root = at_crate_root;
                }
            }
            fn visit_block(&mut self, block: &ast::Block) {
                let at_crate_root = ::std::mem::replace(&mut self.at_crate_root, false);
                visit::walk_block(self, block);
                self.at_crate_root = at_crate_root;
            }
        }

        node.visit_with(&mut MacroLoadingVisitor {
            at_crate_root: self.cx.syntax_env.is_crate_root(),
            cx: self.cx,
        });
    }
}

impl<'a, 'b> Folder for MacroExpander<'a, 'b> {
    fn fold_crate(&mut self, c: Crate) -> Crate {
        self.cx.filename = Some(self.cx.parse_sess.codemap().span_to_filename(c.span));
        noop_fold_crate(c, self)
    }

    fn fold_expr(&mut self, expr: P<ast::Expr>) -> P<ast::Expr> {
        expr.and_then(|expr| expand_expr(expr, self))
    }

    fn fold_opt_expr(&mut self, expr: P<ast::Expr>) -> Option<P<ast::Expr>> {
        expr.and_then(|expr| match expr.node {
            ast::ExprKind::Mac(mac) =>
                expand_mac_invoc(mac, None, expr.attrs.into(), expr.span, self),
            _ => Some(expand_expr(expr, self)),
        })
    }

    fn fold_pat(&mut self, pat: P<ast::Pat>) -> P<ast::Pat> {
        expand_pat(pat, self)
    }

    fn fold_item(&mut self, item: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        use std::mem::replace;
        let result;
        if let ast::ItemKind::Mod(ast::Mod { inner, .. }) = item.node {
            if item.span.contains(inner) {
                self.push_mod_path(item.ident, &item.attrs);
                result = expand_item(item, self);
                self.pop_mod_path();
            } else {
                let filename = if inner != syntax_pos::DUMMY_SP {
                    Some(self.cx.parse_sess.codemap().span_to_filename(inner))
                } else { None };
                let orig_filename = replace(&mut self.cx.filename, filename);
                let orig_mod_path_stack = replace(&mut self.cx.mod_path_stack, Vec::new());
                result = expand_item(item, self);
                self.cx.filename = orig_filename;
                self.cx.mod_path_stack = orig_mod_path_stack;
            }
        } else {
            result = expand_item(item, self);
        }
        result
    }

    fn fold_stmt(&mut self, stmt: ast::Stmt) -> SmallVector<ast::Stmt> {
        expand_stmt(stmt, self)
    }

    fn fold_block(&mut self, block: P<Block>) -> P<Block> {
        let was_in_block = ::std::mem::replace(&mut self.cx.in_block, true);
        let result = with_exts_frame!(self.cx.syntax_env, false, noop_fold_block(block, self));
        self.cx.in_block = was_in_block;
        result
    }

    fn fold_trait_item(&mut self, i: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        expand_annotatable(Annotatable::TraitItem(P(i)), self)
            .into_iter().map(|i| i.expect_trait_item()).collect()
    }

    fn fold_impl_item(&mut self, i: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        expand_annotatable(Annotatable::ImplItem(P(i)), self)
            .into_iter().map(|i| i.expect_impl_item()).collect()
    }

    fn fold_ty(&mut self, ty: P<ast::Ty>) -> P<ast::Ty> {
        expand_type(ty, self)
    }
}

impl<'a, 'b> MacroExpander<'a, 'b> {
    fn push_mod_path(&mut self, id: Ident, attrs: &[ast::Attribute]) {
        let default_path = id.name.as_str();
        let file_path = match ::attr::first_attr_value_str_by_name(attrs, "path") {
            Some(d) => d,
            None => default_path,
        };
        self.cx.mod_path_stack.push(file_path)
    }

    fn pop_mod_path(&mut self) {
        self.cx.mod_path_stack.pop().unwrap();
    }
}

pub struct ExpansionConfig<'feat> {
    pub crate_name: String,
    pub features: Option<&'feat Features>,
    pub recursion_limit: usize,
    pub trace_mac: bool,
    pub should_test: bool, // If false, strip `#[test]` nodes
}

macro_rules! feature_tests {
    ($( fn $getter:ident = $field:ident, )*) => {
        $(
            pub fn $getter(&self) -> bool {
                match self.features {
                    Some(&Features { $field: true, .. }) => true,
                    _ => false,
                }
            }
        )*
    }
}

impl<'feat> ExpansionConfig<'feat> {
    pub fn default(crate_name: String) -> ExpansionConfig<'static> {
        ExpansionConfig {
            crate_name: crate_name,
            features: None,
            recursion_limit: 64,
            trace_mac: false,
            should_test: false,
        }
    }

    feature_tests! {
        fn enable_quotes = quote,
        fn enable_asm = asm,
        fn enable_log_syntax = log_syntax,
        fn enable_concat_idents = concat_idents,
        fn enable_trace_macros = trace_macros,
        fn enable_allow_internal_unstable = allow_internal_unstable,
        fn enable_custom_derive = custom_derive,
        fn enable_pushpop_unsafe = pushpop_unsafe,
        fn enable_rustc_macro = rustc_macro,
    }
}

pub fn expand_crate(cx: &mut ExtCtxt,
                    user_exts: Vec<NamedSyntaxExtension>,
                    c: Crate) -> Crate {
    let mut expander = MacroExpander::new(cx, false, false);
    expand_crate_with_expander(&mut expander, user_exts, c)
}

// Expands crate using supplied MacroExpander - allows for
// non-standard expansion behaviour (e.g. step-wise).
pub fn expand_crate_with_expander(expander: &mut MacroExpander,
                                  user_exts: Vec<NamedSyntaxExtension>,
                                  mut c: Crate) -> Crate {
    if std_inject::no_core(&c) {
        expander.cx.crate_root = None;
    } else if std_inject::no_std(&c) {
        expander.cx.crate_root = Some("core");
    } else {
        expander.cx.crate_root = Some("std");
    }

    // User extensions must be added before expander.load_macros is called,
    // so that macros from external crates shadow user defined extensions.
    for (name, extension) in user_exts {
        expander.cx.syntax_env.insert(name, extension);
    }

    let items = SmallVector::many(c.module.items);
    expander.load_macros(&items);
    c.module.items = items.into();

    let err_count = expander.cx.parse_sess.span_diagnostic.err_count();
    let mut ret = expander.fold_crate(c);
    ret.exported_macros = expander.cx.exported_macros.clone();

    if expander.cx.parse_sess.span_diagnostic.err_count() > err_count {
        expander.cx.parse_sess.span_diagnostic.abort_if_errors();
    }

    ret
}

// A Marker adds the given mark to the syntax context and
// sets spans' `expn_id` to the given expn_id (unless it is `None`).
struct Marker { mark: Mark, expn_id: Option<ExpnId> }

impl Folder for Marker {
    fn fold_ident(&mut self, mut ident: Ident) -> Ident {
        ident.ctxt = ident.ctxt.apply_mark(self.mark);
        ident
    }
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        noop_fold_mac(mac, self)
    }

    fn new_span(&mut self, mut span: Span) -> Span {
        if let Some(expn_id) = self.expn_id {
            span.expn_id = expn_id;
        }
        span
    }
}

// apply a given mark to the given token trees. Used prior to expansion of a macro.
fn mark_tts(tts: &[TokenTree], m: Mark) -> Vec<TokenTree> {
    noop_fold_tts(tts, &mut Marker{mark:m, expn_id: None})
}


#[cfg(test)]
mod tests {
    use super::{expand_crate, ExpansionConfig};
    use ast;
    use ext::base::{ExtCtxt, DummyMacroLoader};
    use parse;
    use util::parser_testing::{string_to_parser};
    use visit;
    use visit::Visitor;

    // a visitor that extracts the paths
    // from a given thingy and puts them in a mutable
    // array (passed in to the traversal)
    #[derive(Clone)]
    struct PathExprFinderContext {
        path_accumulator: Vec<ast::Path> ,
    }

    impl Visitor for PathExprFinderContext {
        fn visit_expr(&mut self, expr: &ast::Expr) {
            if let ast::ExprKind::Path(None, ref p) = expr.node {
                self.path_accumulator.push(p.clone());
            }
            visit::walk_expr(self, expr);
        }
    }

    // these following tests are quite fragile, in that they don't test what
    // *kind* of failure occurs.

    fn test_ecfg() -> ExpansionConfig<'static> {
        ExpansionConfig::default("test".to_string())
    }

    // make sure that macros can't escape fns
    #[should_panic]
    #[test] fn macros_cant_escape_fns_test () {
        let src = "fn bogus() {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        // should fail:
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    // make sure that macros can't escape modules
    #[should_panic]
    #[test] fn macros_cant_escape_mods_test () {
        let src = "mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    // macro_use modules should allow macros to escape
    #[test] fn macros_can_escape_flattened_mods_test () {
        let src = "#[macro_use] mod foo {macro_rules! z (() => (3+4));}\
                   fn inty() -> i32 { z!() }".to_string();
        let sess = parse::ParseSess::new();
        let crate_ast = parse::parse_crate_from_source_str(
            "<test>".to_string(),
            src,
            Vec::new(), &sess).unwrap();
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&sess, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast);
    }

    fn expand_crate_str(crate_str: String) -> ast::Crate {
        let ps = parse::ParseSess::new();
        let crate_ast = panictry!(string_to_parser(&ps, crate_str).parse_crate_mod());
        // the cfg argument actually does matter, here...
        let mut loader = DummyMacroLoader;
        let mut ecx = ExtCtxt::new(&ps, vec![], test_ecfg(), &mut loader);
        expand_crate(&mut ecx, vec![], crate_ast)
    }

    #[test] fn macro_tokens_should_match(){
        expand_crate_str(
            "macro_rules! m((a)=>(13)) ;fn main(){m!(a);}".to_string());
    }

    // should be able to use a bound identifier as a literal in a macro definition:
    #[test] fn self_macro_parsing(){
        expand_crate_str(
            "macro_rules! foo ((zz) => (287;));
            fn f(zz: i32) {foo!(zz);}".to_string()
            );
    }

    // create a really evil test case where a $x appears inside a binding of $x
    // but *shouldn't* bind because it was inserted by a different macro....
    // can't write this test case until we have macro-generating macros.
}
