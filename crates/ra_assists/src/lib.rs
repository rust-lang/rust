//! `ra_assists` crate provides a bunch of code assists, also known as code
//! actions (in LSP) or intentions (in IntelliJ).
//!
//! An assist is a micro-refactoring, which is automatically activated in
//! certain context. For example, if the cursor is over `,`, a "swap `,`" assist
//! becomes available.

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod assist_ctx;
mod marks;
#[cfg(test)]
mod tests;
pub mod utils;
pub mod ast_transform;

use hir::Semantics;
use ra_db::{FileId, FileRange};
use ra_ide_db::{source_change::SourceChange, RootDatabase};
use ra_syntax::TextRange;

pub(crate) use crate::assist_ctx::{Assist, AssistCtx};

/// Unique identifier of the assist, should not be shown to the user
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistId(pub &'static str);

#[derive(Debug, Clone)]
pub struct AssistLabel {
    pub id: AssistId,
    /// Short description of the assist, as shown in the UI.
    pub label: String,
    pub group: Option<GroupLabel>,
    /// Target ranges are used to sort assists: the smaller the target range,
    /// the more specific assist is, and so it should be sorted first.
    pub target: TextRange,
}

#[derive(Clone, Debug)]
pub struct GroupLabel(pub String);

impl AssistLabel {
    pub(crate) fn new(
        id: AssistId,
        label: String,
        group: Option<GroupLabel>,
        target: TextRange,
    ) -> AssistLabel {
        // FIXME: make fields private, so that this invariant can't be broken
        assert!(label.starts_with(|c: char| c.is_uppercase()));
        AssistLabel { id, label, group, target }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedAssist {
    pub label: AssistLabel,
    pub action: SourceChange,
}

#[derive(Debug, Clone, Copy)]
enum AssistFile {
    CurrentFile,
    TargetFile(FileId),
}

impl Default for AssistFile {
    fn default() -> Self {
        Self::CurrentFile
    }
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "unresolved" state, that is only labels are
/// returned, without actual edits.
pub fn unresolved_assists(db: &RootDatabase, range: FileRange) -> Vec<AssistLabel> {
    let sema = Semantics::new(db);
    let ctx = AssistCtx::new(&sema, range, false);
    handlers::all()
        .iter()
        .filter_map(|f| f(ctx.clone()))
        .flat_map(|it| it.0)
        .map(|a| a.label)
        .collect()
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "resolved" state, that is with edit fully
/// computed.
pub fn resolved_assists(db: &RootDatabase, range: FileRange) -> Vec<ResolvedAssist> {
    let sema = Semantics::new(db);
    let ctx = AssistCtx::new(&sema, range, true);
    let mut a = handlers::all()
        .iter()
        .filter_map(|f| f(ctx.clone()))
        .flat_map(|it| it.0)
        .map(|it| it.into_resolved().unwrap())
        .collect::<Vec<_>>();
    a.sort_by_key(|it| it.label.target.len());
    a
}

mod handlers {
    use crate::{Assist, AssistCtx};

    pub(crate) type Handler = fn(AssistCtx) -> Option<Assist>;

    mod add_custom_impl;
    mod add_derive;
    mod add_explicit_type;
    mod add_function;
    mod add_impl;
    mod add_missing_impl_members;
    mod add_new;
    mod apply_demorgan;
    mod auto_import;
    mod change_visibility;
    mod early_return;
    mod fill_match_arms;
    mod flip_binexpr;
    mod flip_comma;
    mod flip_trait_bound;
    mod inline_local_variable;
    mod introduce_variable;
    mod invert_if;
    mod merge_imports;
    mod merge_match_arms;
    mod move_bounds;
    mod move_guard;
    mod raw_string;
    mod remove_dbg;
    mod remove_mut;
    mod replace_if_let_with_match;
    mod replace_let_with_if_let;
    mod replace_qualified_name_with_use;
    mod replace_unwrap_with_match;
    mod split_import;
    mod add_from_impl_for_enum;
    mod reorder_fields;
    mod unwrap_block;

    pub(crate) fn all() -> &'static [Handler] {
        &[
            // These are alphabetic for the foolish consistency
            add_custom_impl::add_custom_impl,
            add_derive::add_derive,
            add_explicit_type::add_explicit_type,
            add_from_impl_for_enum::add_from_impl_for_enum,
            add_function::add_function,
            add_impl::add_impl,
            add_new::add_new,
            apply_demorgan::apply_demorgan,
            auto_import::auto_import,
            change_visibility::change_visibility,
            early_return::convert_to_guarded_return,
            fill_match_arms::fill_match_arms,
            flip_binexpr::flip_binexpr,
            flip_comma::flip_comma,
            flip_trait_bound::flip_trait_bound,
            inline_local_variable::inline_local_variable,
            introduce_variable::introduce_variable,
            invert_if::invert_if,
            merge_imports::merge_imports,
            merge_match_arms::merge_match_arms,
            move_bounds::move_bounds_to_where_clause,
            move_guard::move_arm_cond_to_match_guard,
            move_guard::move_guard_to_arm_body,
            raw_string::add_hash,
            raw_string::make_raw_string,
            raw_string::make_usual_string,
            raw_string::remove_hash,
            remove_dbg::remove_dbg,
            remove_mut::remove_mut,
            reorder_fields::reorder_fields,
            replace_if_let_with_match::replace_if_let_with_match,
            replace_let_with_if_let::replace_let_with_if_let,
            replace_qualified_name_with_use::replace_qualified_name_with_use,
            replace_unwrap_with_match::replace_unwrap_with_match,
            split_import::split_import,
            unwrap_block::unwrap_block,
            // These are manually sorted for better priorities
            add_missing_impl_members::add_missing_impl_members,
            add_missing_impl_members::add_missing_default_members,
            // Are you sure you want to add new assist here, and not to the
            // sorted list above?
        ]
    }
}
