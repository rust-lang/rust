use rustc_abi::ExternAbi;
use rustc_attr_data_structures::{AttributeKind, ReprAttr, find_attr};
use rustc_attr_parsing::AttributeParser;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::FnKind;
use rustc_hir::{AttrArgs, AttrItem, Attribute, GenericParamKind, PatExprKind, PatKind};
use rustc_middle::ty;
use rustc_session::config::CrateType;
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::def_id::LocalDefId;
use rustc_span::{BytePos, Ident, Span, sym};
use {rustc_ast as ast, rustc_hir as hir};

use crate::lints::{
    NonCamelCaseType, NonCamelCaseTypeSub, NonSnakeCaseDiag, NonSnakeCaseDiagSub,
    NonUpperCaseGlobal, NonUpperCaseGlobalSub,
};
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};

#[derive(PartialEq)]
pub(crate) enum MethodLateContext {
    TraitAutoImpl,
    TraitImpl,
    PlainImpl,
}

pub(crate) fn method_context(cx: &LateContext<'_>, id: LocalDefId) -> MethodLateContext {
    let item = cx.tcx.associated_item(id);
    match item.container {
        ty::AssocItemContainer::Trait => MethodLateContext::TraitAutoImpl,
        ty::AssocItemContainer::Impl => match cx.tcx.impl_trait_ref(item.container_id(cx.tcx)) {
            Some(_) => MethodLateContext::TraitImpl,
            None => MethodLateContext::PlainImpl,
        },
    }
}

fn assoc_item_in_trait_impl(cx: &LateContext<'_>, ii: &hir::ImplItem<'_>) -> bool {
    let item = cx.tcx.associated_item(ii.owner_id);
    item.trait_item_def_id.is_some()
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

/// Some unicode characters *have* case, are considered upper case or lower case, but they *can't*
/// be upper cased or lower cased. For the purposes of the lint suggestion, we care about being able
/// to change the char's case.
fn char_has_case(c: char) -> bool {
    let mut l = c.to_lowercase();
    let mut u = c.to_uppercase();
    while let Some(l) = l.next() {
        match u.next() {
            Some(u) if l != u => return true,
            _ => {}
        }
    }
    u.next().is_some()
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
            // be distinguished using an uppercase/lowercase case distinction
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
        let name = ident.name.as_str();

        if !is_camel_case(name) {
            let cc = to_camel_case(name);
            let sub = if *name != cc {
                NonCamelCaseTypeSub::Suggestion { span: ident.span, replace: cc }
            } else {
                NonCamelCaseTypeSub::Label { span: ident.span }
            };
            cx.emit_span_lint(
                NON_CAMEL_CASE_TYPES,
                ident.span,
                NonCamelCaseType { sort, name, sub },
            );
        }
    }
}

impl EarlyLintPass for NonCamelCaseTypes {
    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &ast::Item) {
        let has_repr_c = matches!(
            AttributeParser::parse_limited(cx.sess(), &it.attrs, sym::repr, it.span, it.id),
            Some(Attribute::Parsed(AttributeKind::Repr(r))) if r.iter().any(|(r, _)| r == &ReprAttr::ReprC)
        );

        if has_repr_c {
            return;
        }

        match &it.kind {
            ast::ItemKind::TyAlias(box ast::TyAlias { ident, .. })
            | ast::ItemKind::Enum(ident, ..)
            | ast::ItemKind::Struct(ident, ..)
            | ast::ItemKind::Union(ident, ..) => self.check_case(cx, "type", ident),
            ast::ItemKind::Trait(box ast::Trait { ident, .. }) => {
                self.check_case(cx, "trait", ident)
            }
            ast::ItemKind::TraitAlias(ident, _, _) => self.check_case(cx, "trait alias", ident),

            // N.B. This check is only for inherent associated types, so that we don't lint against
            // trait impls where we should have warned for the trait definition already.
            ast::ItemKind::Impl(box ast::Impl { of_trait: None, items, .. }) => {
                for it in items {
                    // FIXME: this doesn't respect `#[allow(..)]` on the item itself.
                    if let ast::AssocItemKind::Type(alias) = &it.kind {
                        self.check_case(cx, "associated type", &alias.ident);
                    }
                }
            }
            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &EarlyContext<'_>, it: &ast::AssocItem) {
        if let ast::AssocItemKind::Type(alias) = &it.kind {
            self.check_case(cx, "associated type", &alias.ident);
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
    fn to_snake_case(mut name: &str) -> String {
        let mut words = vec![];
        // Preserve leading underscores
        name = name.trim_start_matches(|c: char| {
            if c == '_' {
                words.push(String::new());
                true
            } else {
                false
            }
        });
        for s in name.split('_') {
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

            if ident.contains("__") {
                return false;
            }

            // This correctly handles letters in languages with and without
            // cases, as well as numbers and underscores.
            !ident.chars().any(char::is_uppercase)
        }

        let name = ident.name.as_str();

        if !is_snake_case(name) {
            let span = ident.span;
            let sc = NonSnakeCase::to_snake_case(name);
            // We cannot provide meaningful suggestions
            // if the characters are in the category of "Uppercase Letter".
            let sub = if name != sc {
                // We have a valid span in almost all cases, but we don't have one when linting a
                // crate name provided via the command line.
                if !span.is_dummy() {
                    let sc_ident = Ident::from_str_and_span(&sc, span);
                    if sc_ident.is_reserved() {
                        // We shouldn't suggest a reserved identifier to fix non-snake-case
                        // identifiers. Instead, recommend renaming the identifier entirely or, if
                        // permitted, escaping it to create a raw identifier.
                        if sc_ident.name.can_be_raw() {
                            NonSnakeCaseDiagSub::RenameOrConvertSuggestion {
                                span,
                                suggestion: sc_ident,
                            }
                        } else {
                            NonSnakeCaseDiagSub::SuggestionAndNote { span }
                        }
                    } else {
                        NonSnakeCaseDiagSub::ConvertSuggestion { span, suggestion: sc.clone() }
                    }
                } else {
                    NonSnakeCaseDiagSub::Help
                }
            } else {
                NonSnakeCaseDiagSub::Label { span }
            };
            cx.emit_span_lint(NON_SNAKE_CASE, span, NonSnakeCaseDiag { sort, name, sc, sub });
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonSnakeCase {
    fn check_mod(&mut self, cx: &LateContext<'_>, _: &'tcx hir::Mod<'tcx>, id: hir::HirId) {
        if id != hir::CRATE_HIR_ID {
            return;
        }

        // Issue #45127: don't enforce `snake_case` for binary crates as binaries are not intended
        // to be distributed and depended on like libraries. The lint is not suppressed for cdylib
        // or staticlib because it's not clear what the desired lint behavior for those are.
        if cx.tcx.crate_types().iter().all(|&crate_type| crate_type == CrateType::Executable) {
            return;
        }

        let crate_ident = if let Some(name) = &cx.tcx.sess.opts.crate_name {
            Some(Ident::from_str(name))
        } else {
            ast::attr::find_by_name(cx.tcx.hir_attrs(hir::CRATE_HIR_ID), sym::crate_name).and_then(
                |attr| {
                    if let Attribute::Unparsed(n) = attr
                        && let AttrItem { args: AttrArgs::Eq { eq_span: _, expr: lit }, .. } =
                            n.as_ref()
                        && let ast::LitKind::Str(name, ..) = lit.kind
                    {
                        // Discard the double quotes surrounding the literal.
                        let sp = cx
                            .sess()
                            .source_map()
                            .span_to_snippet(lit.span)
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
                            .unwrap_or(lit.span);

                        Some(Ident::new(name, sp))
                    } else {
                        None
                    }
                },
            )
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
        id: LocalDefId,
    ) {
        match &fk {
            FnKind::Method(ident, sig, ..) => match method_context(cx, id) {
                MethodLateContext::PlainImpl => {
                    if sig.header.abi != ExternAbi::Rust
                        && find_attr!(cx.tcx.get_all_attrs(id), AttributeKind::NoMangle(..))
                    {
                        return;
                    }
                    self.check_snake_case(cx, "method", ident);
                }
                MethodLateContext::TraitAutoImpl => {
                    self.check_snake_case(cx, "trait method", ident);
                }
                _ => (),
            },
            FnKind::ItemFn(ident, _, header) => {
                // Skip foreign-ABI #[no_mangle] functions (Issue #31924)
                if header.abi != ExternAbi::Rust
                    && find_attr!(cx.tcx.get_all_attrs(id), AttributeKind::NoMangle(..))
                {
                    return;
                }
                self.check_snake_case(cx, "function", ident);
            }
            FnKind::Closure => (),
        }
    }

    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        if let hir::ItemKind::Mod(ident, _) = it.kind {
            self.check_snake_case(cx, "module", &ident);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, ty: &hir::Ty<'_, hir::AmbigArg>) {
        if let hir::TyKind::BareFn(hir::BareFnTy { param_idents, .. }) = &ty.kind {
            for param_ident in *param_idents {
                if let Some(param_ident) = param_ident {
                    self.check_snake_case(cx, "variable", param_ident);
                }
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'_>, item: &hir::TraitItem<'_>) {
        if let hir::TraitItemKind::Fn(_, hir::TraitFn::Required(param_idents)) = item.kind {
            self.check_snake_case(cx, "trait method", &item.ident);
            for param_ident in param_idents {
                if let Some(param_ident) = param_ident {
                    self.check_snake_case(cx, "variable", param_ident);
                }
            }
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, p: &hir::Pat<'_>) {
        if let PatKind::Binding(_, hid, ident, _) = p.kind {
            if let hir::Node::PatField(field) = cx.tcx.parent_hir_node(hid) {
                if !field.is_shorthand {
                    // Only check if a new name has been introduced, to avoid warning
                    // on both the struct definition and this pattern.
                    self.check_snake_case(cx, "variable", &ident);
                }
                return;
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
        let name = ident.name.as_str();
        if name.chars().any(|c| c.is_lowercase()) {
            let uc = NonSnakeCase::to_snake_case(name).to_uppercase();
            // We cannot provide meaningful suggestions
            // if the characters are in the category of "Lowercase Letter".
            let sub = if *name != uc {
                NonUpperCaseGlobalSub::Suggestion { span: ident.span, replace: uc }
            } else {
                NonUpperCaseGlobalSub::Label { span: ident.span }
            };
            cx.emit_span_lint(
                NON_UPPER_CASE_GLOBALS,
                ident.span,
                NonUpperCaseGlobal { sort, name, sub },
            );
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for NonUpperCaseGlobals {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &hir::Item<'_>) {
        let attrs = cx.tcx.hir_attrs(it.hir_id());
        match it.kind {
            hir::ItemKind::Static(_, ident, ..)
                if !find_attr!(attrs, AttributeKind::NoMangle(..)) =>
            {
                NonUpperCaseGlobals::check_upper_case(cx, "static variable", &ident);
            }
            hir::ItemKind::Const(ident, ..) => {
                NonUpperCaseGlobals::check_upper_case(cx, "constant", &ident);
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
        if let hir::ImplItemKind::Const(..) = ii.kind
            && !assoc_item_in_trait_impl(cx, ii)
        {
            NonUpperCaseGlobals::check_upper_case(cx, "associated constant", &ii.ident);
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, p: &hir::Pat<'_>) {
        // Lint for constants that look like binding identifiers (#7526)
        if let PatKind::Expr(hir::PatExpr {
            kind: PatExprKind::Path(hir::QPath::Resolved(None, path)),
            ..
        }) = p.kind
        {
            if let Res::Def(DefKind::Const, _) = path.res {
                if let [segment] = path.segments {
                    NonUpperCaseGlobals::check_upper_case(
                        cx,
                        "constant in pattern",
                        &segment.ident,
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
