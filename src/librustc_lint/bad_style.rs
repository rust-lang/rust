// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def::Def;
use rustc::ty;
use lint::{LateContext, LintContext, LintArray};
use lint::{LintPass, LateLintPass};

use syntax::ast;
use syntax::attr;
use syntax_pos::Span;

use rustc::hir::{self, PatKind};
use rustc::hir::intravisit::FnKind;

#[derive(PartialEq)]
pub enum MethodLateContext {
    TraitDefaultImpl,
    TraitImpl,
    PlainImpl,
}

pub fn method_context(cx: &LateContext, id: ast::NodeId, span: Span) -> MethodLateContext {
    let def_id = cx.tcx.map.local_def_id(id);
    match cx.tcx.associated_items.borrow().get(&def_id) {
        None => span_bug!(span, "missing method descriptor?!"),
        Some(item) => {
            match item.container {
                ty::TraitContainer(..) => MethodLateContext::TraitDefaultImpl,
                ty::ImplContainer(cid) => {
                    match cx.tcx.impl_trait_ref(cid) {
                        Some(_) => MethodLateContext::TraitImpl,
                        None => MethodLateContext::PlainImpl,
                    }
                }
            }
        }
    }
}

declare_lint! {
    pub NON_CAMEL_CASE_TYPES,
    Warn,
    "types, variants, traits and type parameters should have camel case names"
}

#[derive(Copy, Clone)]
pub struct NonCamelCaseTypes;

impl NonCamelCaseTypes {
    fn check_case(&self, cx: &LateContext, sort: &str, name: ast::Name, span: Span) {
        fn is_camel_case(name: ast::Name) -> bool {
            let name = name.as_str();
            if name.is_empty() {
                return true;
            }
            let name = name.trim_matches('_');

            // start with a non-lowercase letter rather than non-uppercase
            // ones (some scripts don't have a concept of upper/lowercase)
            !name.is_empty() && !name.chars().next().unwrap().is_lowercase() && !name.contains('_')
        }

        fn to_camel_case(s: &str) -> String {
            s.split('_')
                .flat_map(|word| {
                    word.chars().enumerate().map(|(i, c)| if i == 0 {
                        c.to_uppercase().collect::<String>()
                    } else {
                        c.to_lowercase().collect()
                    })
                })
                .collect::<Vec<_>>()
                .concat()
        }

        if !is_camel_case(name) {
            let c = to_camel_case(&name.as_str());
            let m = if c.is_empty() {
                format!("{} `{}` should have a camel case name such as `CamelCase`", sort, name)
            } else {
                format!("{} `{}` should have a camel case name such as `{}`", sort, name, c)
            };
            cx.span_lint(NON_CAMEL_CASE_TYPES, span, &m[..]);
        }
    }
}

impl LintPass for NonCamelCaseTypes {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_CAMEL_CASE_TYPES)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonCamelCaseTypes {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        let extern_repr_count = it.attrs
            .iter()
            .filter(|attr| {
                attr::find_repr_attrs(cx.tcx.sess.diagnostic(), attr)
                    .iter()
                    .any(|r| r == &attr::ReprExtern)
            })
            .count();
        let has_extern_repr = extern_repr_count > 0;

        if has_extern_repr {
            return;
        }

        match it.node {
            hir::ItemTy(..) |
            hir::ItemStruct(..) |
            hir::ItemUnion(..) => self.check_case(cx, "type", it.name, it.span),
            hir::ItemTrait(..) => self.check_case(cx, "trait", it.name, it.span),
            hir::ItemEnum(ref enum_definition, _) => {
                if has_extern_repr {
                    return;
                }
                self.check_case(cx, "type", it.name, it.span);
                for variant in &enum_definition.variants {
                    self.check_case(cx, "variant", variant.node.name, variant.span);
                }
            }
            _ => (),
        }
    }

    fn check_generics(&mut self, cx: &LateContext, it: &hir::Generics) {
        for gen in it.ty_params.iter() {
            self.check_case(cx, "type parameter", gen.name, gen.span);
        }
    }
}

declare_lint! {
    pub NON_SNAKE_CASE,
    Warn,
    "variables, methods, functions, lifetime parameters and modules should have snake case names"
}

#[derive(Copy, Clone)]
pub struct NonSnakeCase;

impl NonSnakeCase {
    fn to_snake_case(mut str: &str) -> String {
        let mut words = vec![];
        // Preserve leading underscores
        str = str.trim_left_matches(|c: char| {
            if c == '_' {
                words.push(String::new());
                true
            } else {
                false
            }
        });
        for s in str.split('_') {
            let mut last_upper = false;
            let mut buf = String::new();
            if s.is_empty() {
                continue;
            }
            for ch in s.chars() {
                if !buf.is_empty() && buf != "'" && ch.is_uppercase() && !last_upper {
                    words.push(buf);
                    buf = String::new();
                }
                last_upper = ch.is_uppercase();
                buf.extend(ch.to_lowercase());
            }
            words.push(buf);
        }
        words.join("_")
    }

    fn check_snake_case(&self, cx: &LateContext, sort: &str, name: &str, span: Option<Span>) {
        fn is_snake_case(ident: &str) -> bool {
            if ident.is_empty() {
                return true;
            }
            let ident = ident.trim_left_matches('\'');
            let ident = ident.trim_matches('_');

            let mut allow_underscore = true;
            ident.chars().all(|c| {
                allow_underscore = match c {
                    '_' if !allow_underscore => return false,
                    '_' => false,
                    // It would be more obvious to use `c.is_lowercase()`,
                    // but some characters do not have a lowercase form
                    c if !c.is_uppercase() => true,
                    _ => return false,
                };
                true
            })
        }

        if !is_snake_case(name) {
            let sc = NonSnakeCase::to_snake_case(name);
            let msg = if sc != name {
                format!("{} `{}` should have a snake case name such as `{}`",
                        sort,
                        name,
                        sc)
            } else {
                format!("{} `{}` should have a snake case name", sort, name)
            };
            match span {
                Some(span) => cx.span_lint(NON_SNAKE_CASE, span, &msg),
                None => cx.lint(NON_SNAKE_CASE, &msg),
            }
        }
    }
}

impl LintPass for NonSnakeCase {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_SNAKE_CASE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonSnakeCase {
    fn check_crate(&mut self, cx: &LateContext, cr: &hir::Crate) {
        let attr_crate_name = cr.attrs
            .iter()
            .find(|at| at.check_name("crate_name"))
            .and_then(|at| at.value_str().map(|s| (at, s)));
        if let Some(ref name) = cx.tcx.sess.opts.crate_name {
            self.check_snake_case(cx, "crate", name, None);
        } else if let Some((attr, name)) = attr_crate_name {
            self.check_snake_case(cx, "crate", &name.as_str(), Some(attr.span));
        }
    }

    fn check_fn(&mut self,
                cx: &LateContext,
                fk: FnKind,
                _: &hir::FnDecl,
                _: &hir::Body,
                span: Span,
                id: ast::NodeId) {
        match fk {
            FnKind::Method(name, ..) => {
                match method_context(cx, id, span) {
                    MethodLateContext::PlainImpl => {
                        self.check_snake_case(cx, "method", &name.as_str(), Some(span))
                    }
                    MethodLateContext::TraitDefaultImpl => {
                        self.check_snake_case(cx, "trait method", &name.as_str(), Some(span))
                    }
                    _ => (),
                }
            }
            FnKind::ItemFn(name, ..) => {
                self.check_snake_case(cx, "function", &name.as_str(), Some(span))
            }
            FnKind::Closure(_) => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        if let hir::ItemMod(_) = it.node {
            self.check_snake_case(cx, "module", &it.name.as_str(), Some(it.span));
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &hir::TraitItem) {
        if let hir::TraitItemKind::Method(_, hir::TraitMethod::Required(ref names)) = item.node {
            self.check_snake_case(cx,
                                  "trait method",
                                  &item.name.as_str(),
                                  Some(item.span));
            for name in names {
                self.check_snake_case(cx, "variable", &name.node.as_str(), Some(name.span));
            }
        }
    }

    fn check_lifetime_def(&mut self, cx: &LateContext, t: &hir::LifetimeDef) {
        self.check_snake_case(cx,
                              "lifetime",
                              &t.lifetime.name.as_str(),
                              Some(t.lifetime.span));
    }

    fn check_pat(&mut self, cx: &LateContext, p: &hir::Pat) {
        if let &PatKind::Binding(_, _, ref path1, _) = &p.node {
            self.check_snake_case(cx, "variable", &path1.node.as_str(), Some(p.span));
        }
    }

    fn check_struct_def(&mut self,
                        cx: &LateContext,
                        s: &hir::VariantData,
                        _: ast::Name,
                        _: &hir::Generics,
                        _: ast::NodeId) {
        for sf in s.fields() {
            self.check_snake_case(cx, "structure field", &sf.name.as_str(), Some(sf.span));
        }
    }
}

declare_lint! {
    pub NON_UPPER_CASE_GLOBALS,
    Warn,
    "static constants should have uppercase identifiers"
}

#[derive(Copy, Clone)]
pub struct NonUpperCaseGlobals;

impl NonUpperCaseGlobals {
    fn check_upper_case(cx: &LateContext, sort: &str, name: ast::Name, span: Span) {
        if name.as_str().chars().any(|c| c.is_lowercase()) {
            let uc = NonSnakeCase::to_snake_case(&name.as_str()).to_uppercase();
            if name != &*uc {
                cx.span_lint(NON_UPPER_CASE_GLOBALS,
                             span,
                             &format!("{} `{}` should have an upper case name such as `{}`",
                                      sort,
                                      name,
                                      uc));
            } else {
                cx.span_lint(NON_UPPER_CASE_GLOBALS,
                             span,
                             &format!("{} `{}` should have an upper case name", sort, name));
            }
        }
    }
}

impl LintPass for NonUpperCaseGlobals {
    fn get_lints(&self) -> LintArray {
        lint_array!(NON_UPPER_CASE_GLOBALS)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonUpperCaseGlobals {
    fn check_item(&mut self, cx: &LateContext, it: &hir::Item) {
        match it.node {
            hir::ItemStatic(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "static variable", it.name, it.span);
            }
            hir::ItemConst(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant", it.name, it.span);
            }
            _ => {}
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, ti: &hir::TraitItem) {
        match ti.node {
            hir::TraitItemKind::Const(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "associated constant", ti.name, ti.span);
            }
            _ => {}
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext, ii: &hir::ImplItem) {
        match ii.node {
            hir::ImplItemKind::Const(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "associated constant", ii.name, ii.span);
            }
            _ => {}
        }
    }

    fn check_pat(&mut self, cx: &LateContext, p: &hir::Pat) {
        // Lint for constants that look like binding identifiers (#7526)
        if let PatKind::Path(hir::QPath::Resolved(None, ref path)) = p.node {
            if path.segments.len() == 1 && path.segments[0].parameters.is_empty() {
                if let Def::Const(..) = path.def {
                    NonUpperCaseGlobals::check_upper_case(cx,
                                                          "constant in pattern",
                                                          path.segments[0].name,
                                                          path.span);
                }
            }
        }
    }
}
