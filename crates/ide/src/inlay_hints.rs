use std::{
    fmt::{self, Write},
    mem::take,
};

use either::Either;
use hir::{
    known, HasVisibility, HirDisplay, HirDisplayError, HirWrite, ModuleDef, ModuleDefId, Semantics,
};
use ide_db::{base_db::FileRange, famous_defs::FamousDefs, RootDatabase};
use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use stdx::never;
use syntax::{
    ast::{self, AstNode},
    match_ast, NodeOrToken, SyntaxNode, TextRange,
};

use crate::{navigation_target::TryToNav, FileId};

mod closing_brace;
mod implicit_static;
mod fn_lifetime_fn;
mod closure_ret;
mod adjustment;
mod chaining;
mod param_name;
mod binding_mode;
mod bind_pat;
mod discriminant;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHintsConfig {
    pub render_colons: bool,
    pub type_hints: bool,
    pub discriminant_hints: DiscriminantHints,
    pub parameter_hints: bool,
    pub chaining_hints: bool,
    pub adjustment_hints: AdjustmentHints,
    pub adjustment_hints_mode: AdjustmentHintsMode,
    pub adjustment_hints_hide_outside_unsafe: bool,
    pub closure_return_type_hints: ClosureReturnTypeHints,
    pub binding_mode_hints: bool,
    pub lifetime_elision_hints: LifetimeElisionHints,
    pub param_names_for_lifetime_elision_hints: bool,
    pub hide_named_constructor_hints: bool,
    pub hide_closure_initialization_hints: bool,
    pub max_length: Option<usize>,
    pub closing_brace_hints_min_lines: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureReturnTypeHints {
    Always,
    WithBlock,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiscriminantHints {
    Always,
    Never,
    Fieldless,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifetimeElisionHints {
    Always,
    SkipTrivial,
    Never,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdjustmentHints {
    Always,
    ReborrowOnly,
    Never,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AdjustmentHintsMode {
    Prefix,
    Postfix,
    PreferPrefix,
    PreferPostfix,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InlayKind {
    BindingMode,
    Chaining,
    ClosingBrace,
    ClosureReturnType,
    GenericParamList,
    Adjustment,
    AdjustmentPostfix,
    Lifetime,
    Parameter,
    Type,
    Discriminant,
    OpeningParenthesis,
    ClosingParenthesis,
}

#[derive(Debug)]
pub struct InlayHint {
    /// The text range this inlay hint applies to.
    pub range: TextRange,
    /// The kind of this inlay hint. This is used to determine side and padding of the hint for
    /// rendering purposes.
    pub kind: InlayKind,
    /// The actual label to show in the inlay hint.
    pub label: InlayHintLabel,
}

impl InlayHint {
    fn closing_paren(range: TextRange) -> InlayHint {
        InlayHint { range, kind: InlayKind::ClosingParenthesis, label: InlayHintLabel::from(")") }
    }
    fn opening_paren(range: TextRange) -> InlayHint {
        InlayHint { range, kind: InlayKind::OpeningParenthesis, label: InlayHintLabel::from("(") }
    }
}

#[derive(Debug)]
pub enum InlayTooltip {
    String(String),
    Markdown(String),
}

#[derive(Default)]
pub struct InlayHintLabel {
    pub parts: SmallVec<[InlayHintLabelPart; 1]>,
}

impl InlayHintLabel {
    pub fn simple(
        s: impl Into<String>,
        tooltip: Option<InlayTooltip>,
        linked_location: Option<FileRange>,
    ) -> InlayHintLabel {
        InlayHintLabel {
            parts: smallvec![InlayHintLabelPart { text: s.into(), linked_location, tooltip }],
        }
    }

    pub fn prepend_str(&mut self, s: &str) {
        match &mut *self.parts {
            [InlayHintLabelPart { text, linked_location: None, tooltip: None }, ..] => {
                text.insert_str(0, s)
            }
            _ => self.parts.insert(
                0,
                InlayHintLabelPart { text: s.into(), linked_location: None, tooltip: None },
            ),
        }
    }

    pub fn append_str(&mut self, s: &str) {
        match &mut *self.parts {
            [.., InlayHintLabelPart { text, linked_location: None, tooltip: None }] => {
                text.push_str(s)
            }
            _ => self.parts.push(InlayHintLabelPart {
                text: s.into(),
                linked_location: None,
                tooltip: None,
            }),
        }
    }
}

impl From<String> for InlayHintLabel {
    fn from(s: String) -> Self {
        Self {
            parts: smallvec![InlayHintLabelPart { text: s, linked_location: None, tooltip: None }],
        }
    }
}

impl From<&str> for InlayHintLabel {
    fn from(s: &str) -> Self {
        Self {
            parts: smallvec![InlayHintLabelPart {
                text: s.into(),
                linked_location: None,
                tooltip: None
            }],
        }
    }
}

impl fmt::Display for InlayHintLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.parts.iter().map(|part| &part.text).format(""))
    }
}

impl fmt::Debug for InlayHintLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.parts).finish()
    }
}

pub struct InlayHintLabelPart {
    pub text: String,
    /// Source location represented by this label part. The client will use this to fetch the part's
    /// hover tooltip, and Ctrl+Clicking the label part will navigate to the definition the location
    /// refers to (not necessarily the location itself).
    /// When setting this, no tooltip must be set on the containing hint, or VS Code will display
    /// them both.
    pub linked_location: Option<FileRange>,
    /// The tooltip to show when hovering over the inlay hint, this may invoke other actions like
    /// hover requests to show.
    pub tooltip: Option<InlayTooltip>,
}

impl fmt::Debug for InlayHintLabelPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self { text, linked_location: None, tooltip: None } => text.fmt(f),
            Self { text, linked_location, tooltip } => f
                .debug_struct("InlayHintLabelPart")
                .field("text", text)
                .field("linked_location", linked_location)
                .field(
                    "tooltip",
                    &tooltip.as_ref().map_or("", |it| match it {
                        InlayTooltip::String(it) | InlayTooltip::Markdown(it) => it,
                    }),
                )
                .finish(),
        }
    }
}

#[derive(Debug)]
struct InlayHintLabelBuilder<'a> {
    db: &'a RootDatabase,
    result: InlayHintLabel,
    last_part: String,
    location: Option<FileRange>,
}

impl fmt::Write for InlayHintLabelBuilder<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.last_part.write_str(s)
    }
}

impl HirWrite for InlayHintLabelBuilder<'_> {
    fn start_location_link(&mut self, def: ModuleDefId) {
        if self.location.is_some() {
            never!("location link is already started");
        }
        self.make_new_part();
        let Some(location) = ModuleDef::from(def).try_to_nav(self.db) else { return };
        let location =
            FileRange { file_id: location.file_id, range: location.focus_or_full_range() };
        self.location = Some(location);
    }

    fn end_location_link(&mut self) {
        self.make_new_part();
    }
}

impl InlayHintLabelBuilder<'_> {
    fn make_new_part(&mut self) {
        self.result.parts.push(InlayHintLabelPart {
            text: take(&mut self.last_part),
            linked_location: self.location.take(),
            tooltip: None,
        });
    }

    fn finish(mut self) -> InlayHintLabel {
        self.make_new_part();
        self.result
    }
}

fn label_of_ty(
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    ty: hir::Type,
) -> Option<InlayHintLabel> {
    fn rec(
        sema: &Semantics<'_, RootDatabase>,
        famous_defs: &FamousDefs<'_, '_>,
        mut max_length: Option<usize>,
        ty: hir::Type,
        label_builder: &mut InlayHintLabelBuilder<'_>,
    ) -> Result<(), HirDisplayError> {
        let iter_item_type = hint_iterator(sema, famous_defs, &ty);
        match iter_item_type {
            Some((iter_trait, item, ty)) => {
                const LABEL_START: &str = "impl ";
                const LABEL_ITERATOR: &str = "Iterator";
                const LABEL_MIDDLE: &str = "<";
                const LABEL_ITEM: &str = "Item";
                const LABEL_MIDDLE2: &str = " = ";
                const LABEL_END: &str = ">";

                max_length = max_length.map(|len| {
                    len.saturating_sub(
                        LABEL_START.len()
                            + LABEL_ITERATOR.len()
                            + LABEL_MIDDLE.len()
                            + LABEL_MIDDLE2.len()
                            + LABEL_END.len(),
                    )
                });

                label_builder.write_str(LABEL_START)?;
                label_builder.start_location_link(ModuleDef::from(iter_trait).into());
                label_builder.write_str(LABEL_ITERATOR)?;
                label_builder.end_location_link();
                label_builder.write_str(LABEL_MIDDLE)?;
                label_builder.start_location_link(ModuleDef::from(item).into());
                label_builder.write_str(LABEL_ITEM)?;
                label_builder.end_location_link();
                label_builder.write_str(LABEL_MIDDLE2)?;
                rec(sema, famous_defs, max_length, ty, label_builder)?;
                label_builder.write_str(LABEL_END)?;
                Ok(())
            }
            None => ty.display_truncated(sema.db, max_length).write_to(label_builder),
        }
    }

    let mut label_builder = InlayHintLabelBuilder {
        db: sema.db,
        last_part: String::new(),
        location: None,
        result: InlayHintLabel::default(),
    };
    let _ = rec(sema, famous_defs, config.max_length, ty, &mut label_builder);
    let r = label_builder.finish();
    Some(r)
}

// Feature: Inlay Hints
//
// rust-analyzer shows additional information inline with the source code.
// Editors usually render this using read-only virtual text snippets interspersed with code.
//
// rust-analyzer by default shows hints for
//
// * types of local variables
// * names of function arguments
// * types of chained expressions
//
// Optionally, one can enable additional hints for
//
// * return types of closure expressions
// * elided lifetimes
// * compiler inserted reborrows
//
// image::https://user-images.githubusercontent.com/48062697/113020660-b5f98b80-917a-11eb-8d70-3be3fd558cdd.png[]
pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    range_limit: Option<TextRange>,
    config: &InlayHintsConfig,
) -> Vec<InlayHint> {
    let _p = profile::span("inlay_hints");
    let sema = Semantics::new(db);
    let file = sema.parse(file_id);
    let file = file.syntax();

    let mut acc = Vec::new();

    if let Some(scope) = sema.scope(file) {
        let famous_defs = FamousDefs(&sema, scope.krate());

        let hints = |node| hints(&mut acc, &famous_defs, config, file_id, node);
        match range_limit {
            Some(range) => match file.covering_element(range) {
                NodeOrToken::Token(_) => return acc,
                NodeOrToken::Node(n) => n
                    .descendants()
                    .filter(|descendant| range.intersect(descendant.text_range()).is_some())
                    .for_each(hints),
            },
            None => file.descendants().for_each(hints),
        };
    }

    acc
}

fn hints(
    hints: &mut Vec<InlayHint>,
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    file_id: FileId,
    node: SyntaxNode,
) {
    closing_brace::hints(hints, sema, config, file_id, node.clone());
    match_ast! {
        match node {
            ast::Expr(expr) => {
                chaining::hints(hints, famous_defs, config, file_id, &expr);
                adjustment::hints(hints, sema, config, &expr);
                match expr {
                    ast::Expr::CallExpr(it) => param_name::hints(hints, sema, config, ast::Expr::from(it)),
                    ast::Expr::MethodCallExpr(it) => {
                        param_name::hints(hints, sema, config, ast::Expr::from(it))
                    }
                    ast::Expr::ClosureExpr(it) => closure_ret::hints(hints, famous_defs, config, file_id, it),
                    // We could show reborrows for all expressions, but usually that is just noise to the user
                    // and the main point here is to show why "moving" a mutable reference doesn't necessarily move it
                    // ast::Expr::PathExpr(_) => reborrow_hints(hints, sema, config, &expr),
                    _ => None,
                }
            },
            ast::Pat(it) => {
                binding_mode::hints(hints, sema, config, &it);
                if let ast::Pat::IdentPat(it) = it {
                    bind_pat::hints(hints, famous_defs, config, file_id, &it);
                }
                Some(())
            },
            ast::Item(it) => match it {
                // FIXME: record impl lifetimes so they aren't being reused in assoc item lifetime inlay hints
                ast::Item::Impl(_) => None,
                ast::Item::Fn(it) => fn_lifetime_fn::hints(hints, config, it),
                // static type elisions
                ast::Item::Static(it) => implicit_static::hints(hints, config, Either::Left(it)),
                ast::Item::Const(it) => implicit_static::hints(hints, config, Either::Right(it)),
                ast::Item::Enum(it) => discriminant::enum_hints(hints, famous_defs, config, file_id, it),
                _ => None,
            },
            // FIXME: fn-ptr type, dyn fn type, and trait object type elisions
            ast::Type(_) => None,
            _ => None,
        }
    };
}

/// Checks if the type is an Iterator from std::iter and returns the iterator trait and the item type of the concrete iterator.
fn hint_iterator(
    sema: &Semantics<'_, RootDatabase>,
    famous_defs: &FamousDefs<'_, '_>,
    ty: &hir::Type,
) -> Option<(hir::Trait, hir::TypeAlias, hir::Type)> {
    let db = sema.db;
    let strukt = ty.strip_references().as_adt()?;
    let krate = strukt.module(db).krate();
    if krate != famous_defs.core()? {
        return None;
    }
    let iter_trait = famous_defs.core_iter_Iterator()?;
    let iter_mod = famous_defs.core_iter()?;

    // Assert that this struct comes from `core::iter`.
    if !(strukt.visibility(db) == hir::Visibility::Public
        && strukt.module(db).path_to_root(db).contains(&iter_mod))
    {
        return None;
    }

    if ty.impls_trait(db, iter_trait, &[]) {
        let assoc_type_item = iter_trait.items(db).into_iter().find_map(|item| match item {
            hir::AssocItem::TypeAlias(alias) if alias.name(db) == known::Item => Some(alias),
            _ => None,
        })?;
        if let Some(ty) = ty.normalize_trait_assoc_type(db, &[], assoc_type_item) {
            return Some((iter_trait, assoc_type_item, ty));
        }
    }

    None
}

fn closure_has_block_body(closure: &ast::ClosureExpr) -> bool {
    matches!(closure.body(), Some(ast::Expr::BlockExpr(_)))
}

#[cfg(test)]
mod tests {
    use expect_test::Expect;
    use itertools::Itertools;
    use test_utils::extract_annotations;

    use crate::inlay_hints::{AdjustmentHints, AdjustmentHintsMode};
    use crate::DiscriminantHints;
    use crate::{fixture, inlay_hints::InlayHintsConfig, LifetimeElisionHints};

    use super::ClosureReturnTypeHints;

    pub(super) const DISABLED_CONFIG: InlayHintsConfig = InlayHintsConfig {
        discriminant_hints: DiscriminantHints::Never,
        render_colons: false,
        type_hints: false,
        parameter_hints: false,
        chaining_hints: false,
        lifetime_elision_hints: LifetimeElisionHints::Never,
        closure_return_type_hints: ClosureReturnTypeHints::Never,
        adjustment_hints: AdjustmentHints::Never,
        adjustment_hints_mode: AdjustmentHintsMode::Prefix,
        adjustment_hints_hide_outside_unsafe: false,
        binding_mode_hints: false,
        hide_named_constructor_hints: false,
        hide_closure_initialization_hints: false,
        param_names_for_lifetime_elision_hints: false,
        max_length: None,
        closing_brace_hints_min_lines: None,
    };
    pub(super) const TEST_CONFIG: InlayHintsConfig = InlayHintsConfig {
        type_hints: true,
        parameter_hints: true,
        chaining_hints: true,
        closure_return_type_hints: ClosureReturnTypeHints::WithBlock,
        binding_mode_hints: true,
        lifetime_elision_hints: LifetimeElisionHints::Always,
        ..DISABLED_CONFIG
    };

    #[track_caller]
    pub(super) fn check(ra_fixture: &str) {
        check_with_config(TEST_CONFIG, ra_fixture);
    }

    #[track_caller]
    pub(super) fn check_with_config(config: InlayHintsConfig, ra_fixture: &str) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let mut expected = extract_annotations(&analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        let actual = inlay_hints
            .into_iter()
            .map(|it| (it.range, it.label.to_string()))
            .sorted_by_key(|(range, _)| range.start())
            .collect::<Vec<_>>();
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual, "\nExpected:\n{expected:#?}\n\nActual:\n{actual:#?}");
    }

    #[track_caller]
    pub(super) fn check_expect(config: InlayHintsConfig, ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        expect.assert_debug_eq(&inlay_hints)
    }

    #[test]
    fn hints_disabled() {
        check_with_config(
            InlayHintsConfig { render_colons: true, ..DISABLED_CONFIG },
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
}"#,
        );
    }
}
