use std::{
    fmt::{self, Write},
    mem::{self, take},
};

use either::Either;
use hir::{
    ClosureStyle, DisplayTarget, EditionedFileId, HasVisibility, HirDisplay, HirDisplayError,
    HirWrite, InRealFile, ModuleDef, ModuleDefId, Semantics, sym,
};
use ide_db::{FileRange, RootDatabase, famous_defs::FamousDefs, text_edit::TextEditBuilder};
use ide_db::{FxHashSet, text_edit::TextEdit};
use itertools::Itertools;
use smallvec::{SmallVec, smallvec};
use stdx::never;
use syntax::{
    SmolStr, SyntaxNode, TextRange, TextSize, WalkEvent,
    ast::{self, AstNode, HasGenericParams},
    format_smolstr, match_ast,
};

use crate::{FileId, navigation_target::TryToNav};

mod adjustment;
mod bind_pat;
mod binding_mode;
mod bounds;
mod chaining;
mod closing_brace;
mod closure_captures;
mod closure_ret;
mod discriminant;
mod extern_block;
mod generic_param;
mod implicit_drop;
mod implicit_static;
mod implied_dyn_trait;
mod lifetime;
mod param_name;
mod range_exclusive;

// Feature: Inlay Hints
//
// rust-analyzer shows additional information inline with the source code.
// Editors usually render this using read-only virtual text snippets interspersed with code.
//
// rust-analyzer by default shows hints for
//
// * types of local variables
// * names of function arguments
// * names of const generic parameters
// * types of chained expressions
//
// Optionally, one can enable additional hints for
//
// * return types of closure expressions
// * elided lifetimes
// * compiler inserted reborrows
// * names of generic type and lifetime parameters
//
// Note: inlay hints for function argument names are heuristically omitted to reduce noise and will not appear if
// any of the
// [following criteria](https://github.com/rust-lang/rust-analyzer/blob/6b8b8ff4c56118ddee6c531cde06add1aad4a6af/crates/ide/src/inlay_hints/param_name.rs#L92-L99)
// are met:
//
// * the parameter name is a suffix of the function's name
// * the argument is a qualified constructing or call expression where the qualifier is an ADT
// * exact argument<->parameter match(ignoring leading underscore) or parameter is a prefix/suffix
//   of argument with _ splitting it off
// * the parameter name starts with `ra_fixture`
// * the parameter name is a
// [well known name](https://github.com/rust-lang/rust-analyzer/blob/6b8b8ff4c56118ddee6c531cde06add1aad4a6af/crates/ide/src/inlay_hints/param_name.rs#L200)
// in a unary function
// * the parameter name is a
// [single character](https://github.com/rust-lang/rust-analyzer/blob/6b8b8ff4c56118ddee6c531cde06add1aad4a6af/crates/ide/src/inlay_hints/param_name.rs#L201)
// in a unary function
//
// ![Inlay hints](https://user-images.githubusercontent.com/48062697/113020660-b5f98b80-917a-11eb-8d70-3be3fd558cdd.png)
pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    range_limit: Option<TextRange>,
    config: &InlayHintsConfig,
) -> Vec<InlayHint> {
    let _p = tracing::info_span!("inlay_hints").entered();
    let sema = Semantics::new(db);
    let file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));
    let file = sema.parse(file_id);
    let file = file.syntax();

    let mut acc = Vec::new();

    let Some(scope) = sema.scope(file) else {
        return acc;
    };
    let famous_defs = FamousDefs(&sema, scope.krate());
    let display_target = famous_defs.1.to_display_target(sema.db);

    let ctx = &mut InlayHintCtx::default();
    let mut hints = |event| {
        if let Some(node) = handle_event(ctx, event) {
            hints(&mut acc, ctx, &famous_defs, config, file_id, display_target, node);
        }
    };
    let mut preorder = file.preorder();
    while let Some(event) = preorder.next() {
        if matches!((&event, range_limit), (WalkEvent::Enter(node), Some(range)) if range.intersect(node.text_range()).is_none())
        {
            preorder.skip_subtree();
            continue;
        }
        hints(event);
    }
    if let Some(range_limit) = range_limit {
        acc.retain(|hint| range_limit.contains_range(hint.range));
    }
    acc
}

#[derive(Default)]
struct InlayHintCtx {
    lifetime_stacks: Vec<Vec<SmolStr>>,
    extern_block_parent: Option<ast::ExternBlock>,
}

pub(crate) fn inlay_hints_resolve(
    db: &RootDatabase,
    file_id: FileId,
    resolve_range: TextRange,
    hash: u64,
    config: &InlayHintsConfig,
    hasher: impl Fn(&InlayHint) -> u64,
) -> Option<InlayHint> {
    let _p = tracing::info_span!("inlay_hints_resolve").entered();
    let sema = Semantics::new(db);
    let file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(db, file_id));
    let file = sema.parse(file_id);
    let file = file.syntax();

    let scope = sema.scope(file)?;
    let famous_defs = FamousDefs(&sema, scope.krate());
    let mut acc = Vec::new();

    let display_target = famous_defs.1.to_display_target(sema.db);

    let ctx = &mut InlayHintCtx::default();
    let mut hints = |event| {
        if let Some(node) = handle_event(ctx, event) {
            hints(&mut acc, ctx, &famous_defs, config, file_id, display_target, node);
        }
    };

    let mut preorder = file.preorder();
    while let Some(event) = preorder.next() {
        // FIXME: This can miss some hints that require the parent of the range to calculate
        if matches!(&event, WalkEvent::Enter(node) if resolve_range.intersect(node.text_range()).is_none())
        {
            preorder.skip_subtree();
            continue;
        }
        hints(event);
    }
    acc.into_iter().find(|hint| hasher(hint) == hash)
}

fn handle_event(ctx: &mut InlayHintCtx, node: WalkEvent<SyntaxNode>) -> Option<SyntaxNode> {
    match node {
        WalkEvent::Enter(node) => {
            if let Some(node) = ast::AnyHasGenericParams::cast(node.clone()) {
                let params = node
                    .generic_param_list()
                    .map(|it| {
                        it.lifetime_params()
                            .filter_map(|it| {
                                it.lifetime().map(|it| format_smolstr!("{}", &it.text()[1..]))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                ctx.lifetime_stacks.push(params);
            }
            if let Some(node) = ast::ExternBlock::cast(node.clone()) {
                ctx.extern_block_parent = Some(node);
            }
            Some(node)
        }
        WalkEvent::Leave(n) => {
            if ast::AnyHasGenericParams::can_cast(n.kind()) {
                ctx.lifetime_stacks.pop();
            }
            if ast::ExternBlock::can_cast(n.kind()) {
                ctx.extern_block_parent = None;
            }
            None
        }
    }
}

// FIXME: At some point when our hir infra is fleshed out enough we should flip this and traverse the
// HIR instead of the syntax tree.
fn hints(
    hints: &mut Vec<InlayHint>,
    ctx: &mut InlayHintCtx,
    famous_defs @ FamousDefs(sema, _krate): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    file_id: EditionedFileId,
    display_target: DisplayTarget,
    node: SyntaxNode,
) {
    closing_brace::hints(
        hints,
        sema,
        config,
        display_target,
        InRealFile { file_id, value: node.clone() },
    );
    if let Some(any_has_generic_args) = ast::AnyHasGenericArgs::cast(node.clone()) {
        generic_param::hints(hints, famous_defs, config, any_has_generic_args);
    }

    match_ast! {
        match node {
            ast::Expr(expr) => {
                chaining::hints(hints, famous_defs, config, display_target, &expr);
                adjustment::hints(hints, famous_defs, config, display_target, &expr);
                match expr {
                    ast::Expr::CallExpr(it) => param_name::hints(hints, famous_defs, config, ast::Expr::from(it)),
                    ast::Expr::MethodCallExpr(it) => {
                        param_name::hints(hints, famous_defs, config, ast::Expr::from(it))
                    }
                    ast::Expr::ClosureExpr(it) => {
                        closure_captures::hints(hints, famous_defs, config, it.clone());
                        closure_ret::hints(hints, famous_defs, config, display_target, it)
                    },
                    ast::Expr::RangeExpr(it) => range_exclusive::hints(hints, famous_defs, config, it),
                    _ => Some(()),
                }
            },
            ast::Pat(it) => {
                binding_mode::hints(hints, famous_defs, config, &it);
                match it {
                    ast::Pat::IdentPat(it) => {
                        bind_pat::hints(hints, famous_defs, config, display_target, &it);
                    }
                    ast::Pat::RangePat(it) => {
                        range_exclusive::hints(hints, famous_defs, config, it);
                    }
                    _ => {}
                }
                Some(())
            },
            ast::Item(it) => match it {
                ast::Item::Fn(it) => {
                    implicit_drop::hints(hints, famous_defs, config, display_target, &it);
                    if let Some(extern_block) = &ctx.extern_block_parent {
                        extern_block::fn_hints(hints, famous_defs, config, &it, extern_block);
                    }
                    lifetime::fn_hints(hints, ctx, famous_defs, config,  it)
                },
                ast::Item::Static(it) => {
                    if let Some(extern_block) = &ctx.extern_block_parent {
                        extern_block::static_hints(hints, famous_defs, config, &it, extern_block);
                    }
                    implicit_static::hints(hints, famous_defs, config,  Either::Left(it))
                },
                ast::Item::Const(it) => implicit_static::hints(hints, famous_defs, config, Either::Right(it)),
                ast::Item::Enum(it) => discriminant::enum_hints(hints, famous_defs, config, it),
                ast::Item::ExternBlock(it) => extern_block::extern_block_hints(hints, famous_defs, config, it),
                _ => None,
            },
            // FIXME: trait object type elisions
            ast::Type(ty) => match ty {
                ast::Type::FnPtrType(ptr) => lifetime::fn_ptr_hints(hints, ctx, famous_defs, config,  ptr),
                ast::Type::PathType(path) => {
                    lifetime::fn_path_hints(hints, ctx, famous_defs, config, &path);
                    implied_dyn_trait::hints(hints, famous_defs, config, Either::Left(path));
                    Some(())
                },
                ast::Type::DynTraitType(dyn_) => {
                    implied_dyn_trait::hints(hints, famous_defs, config, Either::Right(dyn_));
                    Some(())
                },
                _ => Some(()),
            },
            ast::GenericParamList(it) => bounds::hints(hints, famous_defs, config,  it),
            _ => Some(()),
        }
    };
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHintsConfig {
    pub render_colons: bool,
    pub type_hints: bool,
    pub sized_bound: bool,
    pub discriminant_hints: DiscriminantHints,
    pub parameter_hints: bool,
    pub generic_parameter_hints: GenericParameterHints,
    pub chaining_hints: bool,
    pub adjustment_hints: AdjustmentHints,
    pub adjustment_hints_mode: AdjustmentHintsMode,
    pub adjustment_hints_hide_outside_unsafe: bool,
    pub closure_return_type_hints: ClosureReturnTypeHints,
    pub closure_capture_hints: bool,
    pub binding_mode_hints: bool,
    pub implicit_drop_hints: bool,
    pub lifetime_elision_hints: LifetimeElisionHints,
    pub param_names_for_lifetime_elision_hints: bool,
    pub hide_named_constructor_hints: bool,
    pub hide_closure_initialization_hints: bool,
    pub hide_closure_parameter_hints: bool,
    pub range_exclusive_hints: bool,
    pub closure_style: ClosureStyle,
    pub max_length: Option<usize>,
    pub closing_brace_hints_min_lines: Option<usize>,
    pub fields_to_resolve: InlayFieldsToResolve,
}

impl InlayHintsConfig {
    fn lazy_text_edit(&self, finish: impl FnOnce() -> TextEdit) -> LazyProperty<TextEdit> {
        if self.fields_to_resolve.resolve_text_edits {
            LazyProperty::Lazy
        } else {
            let edit = finish();
            never!(edit.is_empty(), "inlay hint produced an empty text edit");
            LazyProperty::Computed(edit)
        }
    }

    fn lazy_tooltip(&self, finish: impl FnOnce() -> InlayTooltip) -> LazyProperty<InlayTooltip> {
        if self.fields_to_resolve.resolve_hint_tooltip
            && self.fields_to_resolve.resolve_label_tooltip
        {
            LazyProperty::Lazy
        } else {
            let tooltip = finish();
            never!(
                match &tooltip {
                    InlayTooltip::String(s) => s,
                    InlayTooltip::Markdown(s) => s,
                }
                .is_empty(),
                "inlay hint produced an empty tooltip"
            );
            LazyProperty::Computed(tooltip)
        }
    }

    /// This always reports a resolvable location, so only use this when it is very likely for a
    /// location link to actually resolve but where computing `finish` would be costly.
    fn lazy_location_opt(
        &self,
        finish: impl FnOnce() -> Option<FileRange>,
    ) -> Option<LazyProperty<FileRange>> {
        if self.fields_to_resolve.resolve_label_location {
            Some(LazyProperty::Lazy)
        } else {
            finish().map(LazyProperty::Computed)
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InlayFieldsToResolve {
    pub resolve_text_edits: bool,
    pub resolve_hint_tooltip: bool,
    pub resolve_label_tooltip: bool,
    pub resolve_label_location: bool,
    pub resolve_label_command: bool,
}

impl InlayFieldsToResolve {
    pub fn from_client_capabilities(client_capability_fields: &FxHashSet<&str>) -> Self {
        Self {
            resolve_text_edits: client_capability_fields.contains("textEdits"),
            resolve_hint_tooltip: client_capability_fields.contains("tooltip"),
            resolve_label_tooltip: client_capability_fields.contains("label.tooltip"),
            resolve_label_location: client_capability_fields.contains("label.location"),
            resolve_label_command: client_capability_fields.contains("label.command"),
        }
    }

    pub const fn empty() -> Self {
        Self {
            resolve_text_edits: false,
            resolve_hint_tooltip: false,
            resolve_label_tooltip: false,
            resolve_label_location: false,
            resolve_label_command: false,
        }
    }
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
pub struct GenericParameterHints {
    pub type_hints: bool,
    pub lifetime_hints: bool,
    pub const_hints: bool,
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InlayKind {
    Adjustment,
    BindingMode,
    Chaining,
    ClosingBrace,
    ClosureCapture,
    Discriminant,
    GenericParamList,
    Lifetime,
    Parameter,
    GenericParameter,
    Type,
    Dyn,
    Drop,
    RangeExclusive,
    ExternUnsafety,
}

#[derive(Debug, Hash)]
pub enum InlayHintPosition {
    Before,
    After,
}

#[derive(Debug)]
pub struct InlayHint {
    /// The text range this inlay hint applies to.
    pub range: TextRange,
    pub position: InlayHintPosition,
    pub pad_left: bool,
    pub pad_right: bool,
    /// The kind of this inlay hint.
    pub kind: InlayKind,
    /// The actual label to show in the inlay hint.
    pub label: InlayHintLabel,
    /// Text edit to apply when "accepting" this inlay hint.
    pub text_edit: Option<LazyProperty<TextEdit>>,
    /// Range to recompute inlay hints when trying to resolve for this hint. If this is none, the
    /// hint does not support resolving.
    pub resolve_parent: Option<TextRange>,
}

/// A type signaling that a value is either computed, or is available for computation.
#[derive(Clone, Debug)]
pub enum LazyProperty<T> {
    Computed(T),
    Lazy,
}

impl<T> LazyProperty<T> {
    pub fn computed(self) -> Option<T> {
        match self {
            LazyProperty::Computed(it) => Some(it),
            _ => None,
        }
    }

    pub fn is_lazy(&self) -> bool {
        matches!(self, Self::Lazy)
    }
}

impl std::hash::Hash for InlayHint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.range.hash(state);
        self.position.hash(state);
        self.pad_left.hash(state);
        self.pad_right.hash(state);
        self.kind.hash(state);
        self.label.hash(state);
        mem::discriminant(&self.text_edit).hash(state);
    }
}

impl InlayHint {
    fn closing_paren_after(kind: InlayKind, range: TextRange) -> InlayHint {
        InlayHint {
            range,
            kind,
            label: InlayHintLabel::from(")"),
            text_edit: None,
            position: InlayHintPosition::After,
            pad_left: false,
            pad_right: false,
            resolve_parent: None,
        }
    }
}

#[derive(Debug, Hash)]
pub enum InlayTooltip {
    String(String),
    Markdown(String),
}

#[derive(Default, Hash)]
pub struct InlayHintLabel {
    pub parts: SmallVec<[InlayHintLabelPart; 1]>,
}

impl InlayHintLabel {
    pub fn simple(
        s: impl Into<String>,
        tooltip: Option<LazyProperty<InlayTooltip>>,
        linked_location: Option<LazyProperty<FileRange>>,
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

    pub fn append_part(&mut self, part: InlayHintLabelPart) {
        if part.linked_location.is_none() && part.tooltip.is_none() {
            if let Some(InlayHintLabelPart { text, linked_location: None, tooltip: None }) =
                self.parts.last_mut()
            {
                text.push_str(&part.text);
                return;
            }
        }
        self.parts.push(part);
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
    pub linked_location: Option<LazyProperty<FileRange>>,
    /// The tooltip to show when hovering over the inlay hint, this may invoke other actions like
    /// hover requests to show.
    pub tooltip: Option<LazyProperty<InlayTooltip>>,
}

impl std::hash::Hash for InlayHintLabelPart {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.linked_location.is_some().hash(state);
        self.tooltip.is_some().hash(state);
    }
}

impl fmt::Debug for InlayHintLabelPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self { text, linked_location: None, tooltip: None | Some(LazyProperty::Lazy) } => {
                text.fmt(f)
            }
            Self { text, linked_location, tooltip } => f
                .debug_struct("InlayHintLabelPart")
                .field("text", text)
                .field("linked_location", linked_location)
                .field(
                    "tooltip",
                    &tooltip.as_ref().map_or("", |it| match it {
                        LazyProperty::Computed(
                            InlayTooltip::String(it) | InlayTooltip::Markdown(it),
                        ) => it,
                        LazyProperty::Lazy => "",
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
    resolve: bool,
    location: Option<LazyProperty<FileRange>>,
}

impl fmt::Write for InlayHintLabelBuilder<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.last_part.write_str(s)
    }
}

impl HirWrite for InlayHintLabelBuilder<'_> {
    fn start_location_link(&mut self, def: ModuleDefId) {
        never!(self.location.is_some(), "location link is already started");
        self.make_new_part();

        self.location = Some(if self.resolve {
            LazyProperty::Lazy
        } else {
            LazyProperty::Computed({
                let Some(location) = ModuleDef::from(def).try_to_nav(self.db) else { return };
                let location = location.call_site();
                FileRange { file_id: location.file_id, range: location.focus_or_full_range() }
            })
        });
    }

    fn end_location_link(&mut self) {
        self.make_new_part();
    }
}

impl InlayHintLabelBuilder<'_> {
    fn make_new_part(&mut self) {
        let text = take(&mut self.last_part);
        if !text.is_empty() {
            self.result.parts.push(InlayHintLabelPart {
                text,
                linked_location: self.location.take(),
                tooltip: None,
            });
        }
    }

    fn finish(mut self) -> InlayHintLabel {
        self.make_new_part();
        self.result
    }
}

fn label_of_ty(
    famous_defs @ FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    ty: &hir::Type,
    display_target: DisplayTarget,
) -> Option<InlayHintLabel> {
    fn rec(
        sema: &Semantics<'_, RootDatabase>,
        famous_defs: &FamousDefs<'_, '_>,
        mut max_length: Option<usize>,
        ty: &hir::Type,
        label_builder: &mut InlayHintLabelBuilder<'_>,
        config: &InlayHintsConfig,
        display_target: DisplayTarget,
    ) -> Result<(), HirDisplayError> {
        let iter_item_type = hint_iterator(sema, famous_defs, ty);
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
                rec(sema, famous_defs, max_length, &ty, label_builder, config, display_target)?;
                label_builder.write_str(LABEL_END)?;
                Ok(())
            }
            None => ty
                .display_truncated(sema.db, max_length, display_target)
                .with_closure_style(config.closure_style)
                .write_to(label_builder),
        }
    }

    let mut label_builder = InlayHintLabelBuilder {
        db: sema.db,
        last_part: String::new(),
        location: None,
        result: InlayHintLabel::default(),
        resolve: config.fields_to_resolve.resolve_label_location,
    };
    let _ =
        rec(sema, famous_defs, config.max_length, ty, &mut label_builder, config, display_target);
    let r = label_builder.finish();
    Some(r)
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
            hir::AssocItem::TypeAlias(alias) if alias.name(db) == sym::Item => Some(alias),
            _ => None,
        })?;
        if let Some(ty) = ty.normalize_trait_assoc_type(db, &[], assoc_type_item) {
            return Some((iter_trait, assoc_type_item, ty));
        }
    }

    None
}

fn ty_to_text_edit(
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    node_for_hint: &SyntaxNode,
    ty: &hir::Type,
    offset_to_insert_ty: TextSize,
    additional_edits: &dyn Fn(&mut TextEditBuilder),
    prefix: impl Into<String>,
) -> Option<LazyProperty<TextEdit>> {
    // FIXME: Limit the length and bail out on excess somehow?
    let rendered = sema
        .scope(node_for_hint)
        .and_then(|scope| ty.display_source_code(scope.db, scope.module().into(), false).ok())?;
    Some(config.lazy_text_edit(|| {
        let mut builder = TextEdit::builder();
        builder.insert(offset_to_insert_ty, prefix.into());
        builder.insert(offset_to_insert_ty, rendered);

        additional_edits(&mut builder);

        builder.finish()
    }))
}

fn closure_has_block_body(closure: &ast::ClosureExpr) -> bool {
    matches!(closure.body(), Some(ast::Expr::BlockExpr(_)))
}

#[cfg(test)]
mod tests {

    use expect_test::Expect;
    use hir::ClosureStyle;
    use itertools::Itertools;
    use test_utils::extract_annotations;

    use crate::DiscriminantHints;
    use crate::inlay_hints::{AdjustmentHints, AdjustmentHintsMode};
    use crate::{LifetimeElisionHints, fixture, inlay_hints::InlayHintsConfig};

    use super::{ClosureReturnTypeHints, GenericParameterHints, InlayFieldsToResolve};

    pub(super) const DISABLED_CONFIG: InlayHintsConfig = InlayHintsConfig {
        discriminant_hints: DiscriminantHints::Never,
        render_colons: false,
        type_hints: false,
        parameter_hints: false,
        sized_bound: false,
        generic_parameter_hints: GenericParameterHints {
            type_hints: false,
            lifetime_hints: false,
            const_hints: false,
        },
        chaining_hints: false,
        lifetime_elision_hints: LifetimeElisionHints::Never,
        closure_return_type_hints: ClosureReturnTypeHints::Never,
        closure_capture_hints: false,
        adjustment_hints: AdjustmentHints::Never,
        adjustment_hints_mode: AdjustmentHintsMode::Prefix,
        adjustment_hints_hide_outside_unsafe: false,
        binding_mode_hints: false,
        hide_named_constructor_hints: false,
        hide_closure_initialization_hints: false,
        hide_closure_parameter_hints: false,
        closure_style: ClosureStyle::ImplFn,
        param_names_for_lifetime_elision_hints: false,
        max_length: None,
        closing_brace_hints_min_lines: None,
        fields_to_resolve: InlayFieldsToResolve::empty(),
        implicit_drop_hints: false,
        range_exclusive_hints: false,
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
    pub(super) fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(TEST_CONFIG, ra_fixture);
    }

    #[track_caller]
    pub(super) fn check_with_config(
        config: InlayHintsConfig,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let mut expected = extract_annotations(&analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        let actual = inlay_hints
            .into_iter()
            // FIXME: We trim the start because some inlay produces leading whitespace which is not properly supported by our annotation extraction
            .map(|it| (it.range, it.label.to_string().trim_start().to_owned()))
            .sorted_by_key(|(range, _)| range.start())
            .collect::<Vec<_>>();
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual, "\nExpected:\n{expected:#?}\n\nActual:\n{actual:#?}");
    }

    #[track_caller]
    pub(super) fn check_expect(
        config: InlayHintsConfig,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        expect: Expect,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();
        let filtered =
            inlay_hints.into_iter().map(|hint| (hint.range, hint.label)).collect::<Vec<_>>();
        expect.assert_debug_eq(&filtered)
    }

    /// Computes inlay hints for the fixture, applies all the provided text edits and then runs
    /// expect test.
    #[track_caller]
    pub(super) fn check_edit(
        config: InlayHintsConfig,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        expect: Expect,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();

        let edits = inlay_hints
            .into_iter()
            .filter_map(|hint| hint.text_edit?.computed())
            .reduce(|mut acc, next| {
                acc.union(next).expect("merging text edits failed");
                acc
            })
            .expect("no edit returned");

        let mut actual = analysis.file_text(file_id).unwrap().to_string();
        edits.apply(&mut actual);
        expect.assert_eq(&actual);
    }

    #[track_caller]
    pub(super) fn check_no_edit(
        config: InlayHintsConfig,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(&config, file_id, None).unwrap();

        let edits: Vec<_> =
            inlay_hints.into_iter().filter_map(|hint| hint.text_edit?.computed()).collect();

        assert!(edits.is_empty(), "unexpected edits: {edits:?}");
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

    #[test]
    fn regression_18840() {
        check(
            r#"
//- proc_macros: issue_18840
#[proc_macros::issue_18840]
fn foo() {
    let
    loop {}
}
"#,
        );
    }

    #[test]
    fn regression_18898() {
        check(
            r#"
//- proc_macros: issue_18898
#[proc_macros::issue_18898]
fn foo() {
    let
}
"#,
        );
    }

    #[test]
    fn closure_dependency_cycle_no_panic() {
        check(
            r#"
fn foo() {
    let closure;
     // ^^^^^^^ impl Fn()
    closure = || {
        closure();
    };
}

fn bar() {
    let closure1;
     // ^^^^^^^^ impl Fn()
    let closure2;
     // ^^^^^^^^ impl Fn()
    closure1 = || {
        closure2();
    };
    closure2 = || {
        closure1();
    };
}
        "#,
        );
    }

    #[test]
    fn regression_19610() {
        check(
            r#"
trait Trait {
    type Assoc;
}
struct Foo<A>(A);
impl<A: Trait<Assoc = impl Trait>> Foo<A> {
    fn foo<'a, 'b>(_: &'a [i32], _: &'b [i32]) {}
}

fn bar() {
    Foo::foo(&[1], &[2]);
}
"#,
        );
    }
}
