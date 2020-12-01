use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast as ast;
use rustc_attr as attr;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{GenericParamKind, PatKind};
use rustc_middle::ty;
use rustc_span::symbol::sym;
use rustc_span::{symbol::Ident, BytePos, Span};
use rustc_target::spec::abi::Abi;

#[derive(PartialEq)]
pub enum MethodLateContext {
    TraitAutoImpl,
    TraitImpl,
    PlainImpl,
}

pub fn method_context(cx: &LateContext<'_>, id: hir::HirId) -> MethodLateContext {
    let def_id = cx.tcx.hir().local_def_id(id);
    let item = cx.tcx.associated_item(def_id);
    match item.container {
        ty::TraitContainer(..) => MethodLateContext::TraitAutoImpl,
        ty::ImplContainer(cid) => match cx.tcx.impl_trait_ref(cid) {
            Some(_) => MethodLateContext::TraitImpl,
            None => MethodLateContext::PlainImpl,
        },
    }
}

declare_lint! {
    /// The `non_camel_case_types` lint detects types, variants, traits and
    /// type parameters that don't have camel case names.
    ///
    /// ### Example
    ///
    /// ```rust
    /// struct my_struct;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred style for these identifiers is to use "camel case", such
    /// as `MyStruct`, where the first letter should not be lowercase, and
    /// should not use underscores between letters. Underscores are allowed at
    /// the beginning and end of the identifier, as well as between
    /// non-letters (such as `X86_64`).
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
        && !name.chars().collect::<Vec<_>>().array_windows().any(|&[fst, snd]| {
            // contains a capitalisable character followed by, or preceded by, an underscore
            char_has_case(fst) && snd == '_' || char_has_case(snd) && fst == '_'
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
                    camel_cased_component.extend(c.to_uppercase());
                } else {
                    camel_cased_component.extend(c.to_lowercase());
                }

                prev_is_lower_case = c.is_lowercase();
                new_word = false;
            }

            camel_cased_component
        })
        .fold((String::new(), None), |(acc, prev): (String, Option<String>), next| {
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
        })
        .0
}

impl NonCamelCaseTypes {
    fn check_case(&self, cx: &EarlyContext<'_>, sort: &str, ident: &Ident) {
        let name = &ident.name.as_str();

        if !is_camel_case(name) {
            cx.struct_span_lint(NON_CAMEL_CASE_TYPES, ident.span, |lint| {
                let msg = format!("{} `{}` should have an upper camel case name", sort, name);
                let mut err = lint.build(&msg);
                let cc = to_camel_case(name);
                // We cannot provide meaningful suggestions
                // if the characters are in the category of "Lowercase Letter".
                if name.to_string() != cc {
                    err.span_suggestion(
                        ident.span,
                        "convert the identifier to upper camel case",
                        to_camel_case(name),
                        Applicability::MaybeIncorrect,
                    );
                }

                err.emit();
            })
        }
    }
}

impl EarlyLintPass for NonCamelCaseTypes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &ast::Item) {
        let has_repr_c = it
            .attrs
            .iter()
            .any(|attr| attr::find_repr_attrs(&cx.sess, attr).contains(&attr::ReprC));

        if has_repr_c {
            return;
        }

        match it.kind {
            ast::ItemKind::TyAlias(..)
            | ast::ItemKind::Enum(..)
            | ast::ItemKind::Struct(..)
            | ast::ItemKind::Union(..) => self.check_case(cx, "type", &it.ident),
            ast::ItemKind::Trait(..) => self.check_case(cx, "trait", &it.ident),
            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &EarlyContext<'_>, it: &ast::AssocItem) {
        if let ast::AssocItemKind::TyAlias(..) = it.kind {
            self.check_case(cx, "associated type", &it.ident);
        }
    }

    fn check_variant(&mut self, cx: &EarlyContext<'_>, v: &ast::Variant) {
        self.check_case(cx, "variant", &v.ident);
    }

    fn check_generic_param(&mut self, cx: &EarlyContext<'_>, param: &ast::GenericParam) {
        if let ast::GenericParamKind::Type { .. } = param.kind {
            self.check_case(cx, "type parameter", &param.ident);
        }
    }
}

declare_lint! {
    /// The `non_snake_case` lint detects variables, methods, functions,
    /// lifetime parameters and modules that don't have snake case names.
    ///
    /// ### Example
    ///
    /// ```rust
    /// let MY_VALUE = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred style for these identifiers is to use "snake case",
    /// where all the characters are in lowercase, with words separated with a
    /// single underscore, such as `my_value`.
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
    fn check_snake_case(&self, cx: &LateContext<'_>, sort: &str, ident: &Ident) {
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
            cx.struct_span_lint(NON_SNAKE_CASE, ident.span, |lint| {
                let sc = NonSnakeCase::to_snake_case(name);
                let msg = format!("{} `{}` should have a snake case name", sort, name);
                let mut err = lint.build(&msg);
                // We cannot provide meaningful suggestions
                // if the characters are in the category of "Uppercase Letter".
                if name.to_string() != sc {
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
                }

                err.emit();
            });
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonSnakeCase {
    fn check_mod(
        &mut self,
        cx: &LateContext<'_>,
        _: &'tcx hir::Mod<'tcx>,
        _: Span,
        id: hir::HirId,
    ) {
        if id != hir::CRATE_HIR_ID {
            return;
        }

        let crate_ident = if let Some(name) = &cx.tcx.sess.opts.crate_name {
            Some(Ident::from_str(name))
        } else {
            cx.sess()
                .find_by_name(&cx.tcx.hir().attrs(hir::CRATE_HIR_ID), sym::crate_name)
                .and_then(|attr| attr.meta())
                .and_then(|meta| {
                    meta.name_value_literal().and_then(|lit| {
                        if let ast::LitKind::Str(name, ..) = lit.kind {
                            // Discard the double quotes surrounding the literal.
                            let sp = cx
                                .sess()
                                .source_map()
                                .span_to_snippet(lit.span)
                                .ok()
                                .and_then(|snippet| {
                                    let left = snippet.find('"')?;
                                    let right =
                                        snippet.rfind('"').map(|pos| snippet.len() - pos)?;

                                    Some(
                                        lit.span
                                            .with_lo(lit.span.lo() + BytePos(left as u32 + 1))
                                            .with_hi(lit.span.hi() - BytePos(right as u32)),
                                    )
                                })
                                .unwrap_or(lit.span);

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

    fn check_generic_param(&mut self, cx: &LateContext<'_>, param: &hir::GenericParam<'_>) {
        if let GenericParamKind::Lifetime { .. } = param.kind {
            self.check_snake_case(cx, "lifetime", &param.name.ident());
        }
    }

    fn check_fn(
        &mut self,
        cx: &LateContext<'_>,
        fk: FnKind<'_>,
        _: &hir::FnDecl<'_>,
        _: &hir::Body<'_>,
        _: Span,
        id: hir::HirId,
    ) {
        match &fk {
            FnKind::Method(ident, ..) => match method_context(cx, id) {
                MethodLateContext::PlainImpl => {
                    self.check_snake_case(cx, "method", ident);
                }
                MethodLateContext::TraitAutoImpl => {
                    self.check_snake_case(cx, "trait method", ident);
                }
                _ => (),
            },
            FnKind::ItemFn(ident, _, header, _, attrs) => {
                // Skip foreign-ABI #[no_mangle] functions (Issue #31924)
                if header.abi != Abi::Rust && cx.sess().contains_name(attrs, sym::no_mangle) {
                    return;
                }
                self.check_snake_case(cx, "function", ident);
            }
            FnKind::Closure(_) => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Mod(_) = it.kind {
            self.check_snake_case(cx, "module", &it.ident);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &hir::TraitItem<'_>) {
        if let hir::TraitItemKind::Fn(_, hir::TraitFn::Required(pnames)) = item.kind {
            self.check_snake_case(cx, "trait method", &item.ident);
            for param_name in pnames {
                self.check_snake_case(cx, "variable", param_name);
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, p: &hir::Pat<'_>) {
        if let &PatKind::Binding(_, hid, ident, _) = &p.kind {
            if let hir::Node::Pat(parent_pat) = cx.tcx.hir().get(cx.tcx.hir().get_parent_node(hid))
            {
                if let PatKind::Struct(_, field_pats, _) = &parent_pat.kind {
                    for field in field_pats.iter() {
                        if field.ident != ident {
                            // Only check if a new name has been introduced, to avoid warning
                            // on both the struct definition and this pattern.
                            self.check_snake_case(cx, "variable", &ident);
                        }
                    }
                    return;
                }
            }
            self.check_snake_case(cx, "variable", &ident);
        }
    }

    fn check_struct_def(&mut self, cx: &LateContext<'_>, s: &hir::VariantData<'_>) {
        for sf in s.fields() {
            self.check_snake_case(cx, "structure field", &sf.ident);
        }
    }
}

declare_lint! {
    /// The `non_upper_case_globals` lint detects static items that don't have
    /// uppercase identifiers.
    ///
    /// ### Example
    ///
    /// ```rust
    /// static max_points: i32 = 5;
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// The preferred style is for static item names to use all uppercase
    /// letters such as `MAX_POINTS`.
    pub NON_UPPER_CASE_GLOBALS,
    Warn,
    "static constants should have uppercase identifiers"
}

declare_lint_pass!(NonUpperCaseGlobals => [NON_UPPER_CASE_GLOBALS]);

impl NonUpperCaseGlobals {
    fn check_upper_case(cx: &LateContext<'_>, sort: &str, ident: &Ident) {
        let name = &ident.name.as_str();
        if name.chars().any(|c| c.is_lowercase()) {
            cx.struct_span_lint(NON_UPPER_CASE_GLOBALS, ident.span, |lint| {
                let uc = NonSnakeCase::to_snake_case(&name).to_uppercase();
                let mut err =
                    lint.build(&format!("{} `{}` should have an upper case name", sort, name));
                // We cannot provide meaningful suggestions
                // if the characters are in the category of "Lowercase Letter".
                if name.to_string() != uc {
                    err.span_suggestion(
                        ident.span,
                        "convert the identifier to upper case",
                        uc,
                        Applicability::MaybeIncorrect,
                    );
                }

                err.emit();
            })
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonUpperCaseGlobals {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        match it.kind {
            hir::ItemKind::Static(..) if !cx.sess().contains_name(&it.attrs, sym::no_mangle) => {
                NonUpperCaseGlobals::check_upper_case(cx, "static variable", &it.ident);
            }
            hir::ItemKind::Const(..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant", &it.ident);
            }
            _ => {}
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, ti: &hir::TraitItem<'_>) {
        if let hir::TraitItemKind::Const(..) = ti.kind {
            NonUpperCaseGlobals::check_upper_case(cx, "associated constant", &ti.ident);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, ii: &hir::ImplItem<'_>) {
        if let hir::ImplItemKind::Const(..) = ii.kind {
            NonUpperCaseGlobals::check_upper_case(cx, "associated constant", &ii.ident);
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, p: &hir::Pat<'_>) {
        // Lint for constants that look like binding identifiers (#7526)
        if let PatKind::Path(hir::QPath::Resolved(None, ref path)) = p.kind {
            if let Res::Def(DefKind::Const, _) = path.res {
                if path.segments.len() == 1 {
                    NonUpperCaseGlobals::check_upper_case(
                        cx,
                        "constant in pattern",
                        &path.segments[0].ident,
                    );
                }
            }
        }
    }

    fn check_generic_param(&mut self, cx: &LateContext<'_>, param: &hir::GenericParam<'_>) {
        if let GenericParamKind::Const { .. } = param.kind {
            NonUpperCaseGlobals::check_upper_case(cx, "const parameter", &param.name.ident());
        }
    }
}

#[cfg(test)]
mod tests;
