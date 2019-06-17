use rustc::hir::{self, GenericParamKind, PatKind};
use rustc::hir::def::{Res, DefKind};
use rustc::hir::intravisit::FnKind;
use rustc::lint;
use rustc::ty;
use rustc_target::spec::abi::Abi;
use lint::{EarlyContext, LateContext, LintContext, LintArray};
use lint::{EarlyLintPass, LintPass, LateLintPass};
use syntax::ast;
use syntax::attr;
use syntax::errors::Applicability;
use syntax::symbol::sym;
use syntax_pos::{BytePos, symbol::Ident, Span};

#[derive(PartialEq)]
pub enum MethodLateContext {
    TraitAutoImpl,
    TraitImpl,
    PlainImpl,
}

pub fn method_context(cx: &LateContext<'_, '_>, id: hir::HirId) -> MethodLateContext {
    let def_id = cx.tcx.hir().local_def_id_from_hir_id(id);
    let item = cx.tcx.associated_item(def_id);
    match item.container {
        ty::TraitContainer(..) => MethodLateContext::TraitAutoImpl,
        ty::ImplContainer(cid) => {
            match cx.tcx.impl_trait_ref(cid) {
                Some(_) => MethodLateContext::TraitImpl,
                None => MethodLateContext::PlainImpl,
            }
        }
    }
}

declare_lint! {
    pub NON_CAMEL_CASE_TYPES,
    Warn,
    "types, variants, traits and type parameters should have camel case names"
}

declare_lint_pass!(NonCamelCaseTypes => [NON_CAMEL_CASE_TYPES]);

fn char_has_case(c: char) -> bool {
    c.is_lowercase() || c.is_uppercase()
}

fn is_camel_case(name: &str) -> bool {
    let name = name.trim_matches('_');
    if name.is_empty() {
        return true;
    }

    // start with a non-lowercase letter rather than non-uppercase
    // ones (some scripts don't have a concept of upper/lowercase)
    !name.chars().next().unwrap().is_lowercase()
        && !name.contains("__")
        && !name.chars().collect::<Vec<_>>().windows(2).any(|pair| {
            // contains a capitalisable character followed by, or preceded by, an underscore
            char_has_case(pair[0]) && pair[1] == '_' || char_has_case(pair[1]) && pair[0] == '_'
        })
}

fn to_camel_case(s: &str) -> String {
    s.trim_matches('_')
        .split('_')
        .filter(|component| !component.is_empty())
        .map(|component| {
            let mut camel_cased_component = String::new();

            let mut new_word = true;
            let mut prev_is_lower_case = true;

            for c in component.chars() {
                // Preserve the case if an uppercase letter follows a lowercase letter, so that
                // `camelCase` is converted to `CamelCase`.
                if prev_is_lower_case && c.is_uppercase() {
                    new_word = true;
                }

                if new_word {
                    camel_cased_component.push_str(&c.to_uppercase().to_string());
                } else {
                    camel_cased_component.push_str(&c.to_lowercase().to_string());
                }

                prev_is_lower_case = c.is_lowercase();
                new_word = false;
            }

            camel_cased_component
        })
        .fold(
            (String::new(), None),
            |(acc, prev): (String, Option<String>), next| {
                // separate two components with an underscore if their boundary cannot
                // be distinguished using a uppercase/lowercase case distinction
                let join = if let Some(prev) = prev {
                    let l = prev.chars().last().unwrap();
                    let f = next.chars().next().unwrap();
                    !char_has_case(l) && !char_has_case(f)
                } else {
                    false
                };
                (acc + if join { "_" } else { "" } + &next, Some(next))
            },
        )
        .0
}

impl NonCamelCaseTypes {
    fn check_case(&self, cx: &EarlyContext<'_>, sort: &str, ident: &Ident) {
        let name = &ident.name.as_str();

        if !is_camel_case(name) {
            let msg = format!("{} `{}` should have an upper camel case name", sort, name);
            cx.struct_span_lint(NON_CAMEL_CASE_TYPES, ident.span, &msg)
                .span_suggestion(
                    ident.span,
                    "convert the identifier to upper camel case",
                    to_camel_case(name),
                    Applicability::MaybeIncorrect,
                )
                .emit();
        }
    }
}

impl EarlyLintPass for NonCamelCaseTypes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &ast::Item) {
        let has_repr_c = it.attrs
            .iter()
            .any(|attr| attr::find_repr_attrs(&cx.sess.parse_sess, attr).contains(&attr::ReprC));

        if has_repr_c {
            return;
        }

        match it.node {
            ast::ItemKind::Ty(..) |
            ast::ItemKind::Enum(..) |
            ast::ItemKind::Struct(..) |
            ast::ItemKind::Union(..) => self.check_case(cx, "type", &it.ident),
            ast::ItemKind::Trait(..) => self.check_case(cx, "trait", &it.ident),
            _ => (),
        }
    }

    fn check_variant(&mut self, cx: &EarlyContext<'_>, v: &ast::Variant, _: &ast::Generics) {
        self.check_case(cx, "variant", &v.node.ident);
    }

    fn check_generic_param(&mut self, cx: &EarlyContext<'_>, param: &ast::GenericParam) {
        if let ast::GenericParamKind::Type { .. } = param.kind {
            self.check_case(cx, "type parameter", &param.ident);
        }
    }
}

declare_lint! {
    pub NON_SNAKE_CASE,
    Warn,
    "variables, methods, functions, lifetime parameters and modules should have snake case names"
}

declare_lint_pass!(NonSnakeCase => [NON_SNAKE_CASE]);

impl NonSnakeCase {
    fn to_snake_case(mut str: &str) -> String {
        let mut words = vec![];
        // Preserve leading underscores
        str = str.trim_start_matches(|c: char| {
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

    /// Checks if a given identifier is snake case, and reports a diagnostic if not.
    fn check_snake_case(&self, cx: &LateContext<'_, '_>, sort: &str, ident: &Ident) {
        fn is_snake_case(ident: &str) -> bool {
            if ident.is_empty() {
                return true;
            }
            let ident = ident.trim_start_matches('\'');
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

        let name = &ident.name.as_str();

        if !is_snake_case(name) {
            let sc = NonSnakeCase::to_snake_case(name);

            let msg = format!("{} `{}` should have a snake case name", sort, name);
            let mut err = cx.struct_span_lint(NON_SNAKE_CASE, ident.span, &msg);

            // We have a valid span in almost all cases, but we don't have one when linting a crate
            // name provided via the command line.
            if !ident.span.is_dummy() {
                err.span_suggestion(
                    ident.span,
                    "convert the identifier to snake case",
                    sc,
                    Applicability::MaybeIncorrect,
                );
            } else {
                err.help(&format!("convert the identifier to snake case: `{}`", sc));
            }

            err.emit();
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonSnakeCase {
    fn check_mod(&mut self, cx: &LateContext<'_, '_>, _: &'tcx hir::Mod, _: Span, id: hir::HirId) {
        if id != hir::CRATE_HIR_ID {
            return;
        }

        let crate_ident = if let Some(name) = &cx.tcx.sess.opts.crate_name {
            Some(Ident::from_str(name))
        } else {
            attr::find_by_name(&cx.tcx.hir().attrs(hir::CRATE_HIR_ID), sym::crate_name)
                .and_then(|attr| attr.meta())
                .and_then(|meta| {
                    meta.name_value_literal().and_then(|lit| {
                        if let ast::LitKind::Str(name, ..) = lit.node {
                            // Discard the double quotes surrounding the literal.
                            let sp = cx.sess().source_map().span_to_snippet(lit.span)
                                .ok()
                                .and_then(|snippet| {
                                    let left = snippet.find('"')?;
                                    let right = snippet.rfind('"').map(|pos| snippet.len() - pos)?;

                                    Some(
                                        lit.span
                                            .with_lo(lit.span.lo() + BytePos(left as u32 + 1))
                                            .with_hi(lit.span.hi() - BytePos(right as u32)),
                                    )
                                })
                                .unwrap_or_else(|| lit.span);

                            Some(Ident::new(name, sp))
                        } else {
                            None
                        }
                    })
                })
        };

        if let Some(ident) = &crate_ident {
            self.check_snake_case(cx, "crate", ident);
        }
    }

    fn check_generic_param(&mut self, cx: &LateContext<'_, '_>, param: &hir::GenericParam) {
        if let GenericParamKind::Lifetime { .. } = param.kind {
            self.check_snake_case(cx, "lifetime", &param.name.ident());
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'_, '_>,
        fk: FnKind<'_>,
        _: &hir::FnDecl,
        _: &hir::Body,
        _: Span,
        id: hir::HirId,
    ) {
        match &fk {
            FnKind::Method(ident, ..) => {
                match method_context(cx, id) {
                    MethodLateContext::PlainImpl => {
                        self.check_snake_case(cx, "method", ident);
                    }
                    MethodLateContext::TraitAutoImpl => {
                        self.check_snake_case(cx, "trait method", ident);
                    }
                    _ => (),
                }
            }
            FnKind::ItemFn(ident, _, header, _, attrs) => {
                // Skip foreign-ABI #[no_mangle] functions (Issue #31924)
                if header.abi != Abi::Rust && attr::contains_name(attrs, sym::no_mangle) {
                    return;
                }
                self.check_snake_case(cx, "function", ident);
            }
            FnKind::Closure(_) => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::Item) {
        if let hir::ItemKind::Mod(_) = it.node {
            self.check_snake_case(cx, "module", &it.ident);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_, '_>, item: &hir::TraitItem) {
        if let hir::TraitItemKind::Method(_, hir::TraitMethod::Required(pnames)) = &item.node {
            self.check_snake_case(cx, "trait method", &item.ident);
            for param_name in pnames {
                self.check_snake_case(cx, "variable", param_name);
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_, '_>, p: &hir::Pat) {
        if let &PatKind::Binding(_, _, ident, _) = &p.node {
            self.check_snake_case(cx, "variable", &ident);
        }
    }

    fn check_struct_def(
        &mut self,
        cx: &LateContext<'_, '_>,
        s: &hir::VariantData,
        _: ast::Name,
        _: &hir::Generics,
        _: hir::HirId,
    ) {
        for sf in s.fields() {
            self.check_snake_case(cx, "structure field", &sf.ident);
        }
    }
}

declare_lint! {
    pub NON_UPPER_CASE_GLOBALS,
    Warn,
    "static constants should have uppercase identifiers"
}

declare_lint_pass!(NonUpperCaseGlobals => [NON_UPPER_CASE_GLOBALS]);

impl NonUpperCaseGlobals {
    fn check_upper_case(cx: &LateContext<'_, '_>, sort: &str, ident: &Ident) {
        let name = &ident.name.as_str();

        if name.chars().any(|c| c.is_lowercase()) {
            let uc = NonSnakeCase::to_snake_case(&name).to_uppercase();

            let msg = format!("{} `{}` should have an upper case name", sort, name);
            cx.struct_span_lint(NON_UPPER_CASE_GLOBALS, ident.span, &msg)
                .span_suggestion(
                    ident.span,
                    "convert the identifier to upper case",
                    uc,
                    Applicability::MaybeIncorrect,
                )
                .emit();
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NonUpperCaseGlobals {
    fn check_item(&mut self, cx: &LateContext<'_, '_>, it: &hir::Item) {
        match it.node {
            hir::ItemKind::Static(..) if !attr::contains_name(&it.attrs, sym::no_mangle) => {
                NonUpperCaseGlobals::check_upper_case(cx, "static variable", &it.ident);
            }
            hir::ItemKind::Const(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant", &it.ident);
            }
            _ => {}
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_, '_>, ti: &hir::TraitItem) {
        if let hir::TraitItemKind::Const(..) = ti.node {
            NonUpperCaseGlobals::check_upper_case(cx, "associated constant", &ti.ident);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_, '_>, ii: &hir::ImplItem) {
        if let hir::ImplItemKind::Const(..) = ii.node {
            NonUpperCaseGlobals::check_upper_case(cx, "associated constant", &ii.ident);
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_, '_>, p: &hir::Pat) {
        // Lint for constants that look like binding identifiers (#7526)
        if let PatKind::Path(hir::QPath::Resolved(None, ref path)) = p.node {
            if let Res::Def(DefKind::Const, _) = path.res {
                if path.segments.len() == 1 {
                    NonUpperCaseGlobals::check_upper_case(
                        cx,
                        "constant in pattern",
                        &path.segments[0].ident
                    );
                }
            }
        }
    }

    fn check_generic_param(&mut self, cx: &LateContext<'_, '_>, param: &hir::GenericParam) {
        if let GenericParamKind::Const { .. } = param.kind {
            NonUpperCaseGlobals::check_upper_case(
                cx,
                "const parameter",
                &param.name.ident(),
            );
        }
    }
}

#[cfg(test)]
mod tests;
