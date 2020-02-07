//! `ra_assists` crate provides a bunch of code assists, also known as code
//! actions (in LSP) or intentions (in IntelliJ).
//!
//! An assist is a micro-refactoring, which is automatically activated in
//! certain context. For example, if the cursor is over `,`, a "swap `,`" assist
//! becomes available.

mod assist_ctx;
mod marks;
#[cfg(test)]
mod doc_tests;
pub mod ast_transform;

use std::cmp::Ordering;

use either::Either;
use ra_db::FileRange;
use ra_ide_db::RootDatabase;
use ra_syntax::{TextRange, TextUnit};
use ra_text_edit::TextEdit;

pub(crate) use crate::assist_ctx::{Assist, AssistCtx};
pub use crate::assists::add_import::auto_import_text_edit;

/// Unique identifier of the assist, should not be shown to the user
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistId(pub &'static str);

#[derive(Debug, Clone)]
pub struct AssistLabel {
    /// Short description of the assist, as shown in the UI.
    pub label: String,
    pub id: AssistId,
}

impl AssistLabel {
    pub(crate) fn new(label: String, id: AssistId) -> AssistLabel {
        // FIXME: make fields private, so that this invariant can't be broken
        assert!(label.chars().nth(0).unwrap().is_uppercase());
        AssistLabel { label: label.into(), id }
    }
}

#[derive(Debug, Clone)]
pub struct AssistAction {
    pub label: Option<String>,
    pub edit: TextEdit,
    pub cursor_position: Option<TextUnit>,
    // FIXME: This belongs to `AssistLabel`
    pub target: Option<TextRange>,
}

#[derive(Debug, Clone)]
pub struct ResolvedAssist {
    pub label: AssistLabel,
    pub action_data: Either<AssistAction, Vec<AssistAction>>,
}

impl ResolvedAssist {
    pub fn get_first_action(&self) -> AssistAction {
        match &self.action_data {
            Either::Left(action) => action.clone(),
            Either::Right(actions) => actions[0].clone(),
        }
    }
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "unresolved" state, that is only labels are
/// returned, without actual edits.
pub fn unresolved_assists(db: &RootDatabase, range: FileRange) -> Vec<AssistLabel> {
    let ctx = AssistCtx::new(db, range, false);
    assists::all()
        .iter()
        .filter_map(|f| f(ctx.clone()))
        .map(|a| match a {
            Assist::Unresolved { label } => label,
            Assist::Resolved { .. } => unreachable!(),
        })
        .collect()
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "resolved" state, that is with edit fully
/// computed.
pub fn resolved_assists(db: &RootDatabase, range: FileRange) -> Vec<ResolvedAssist> {
    let ctx = AssistCtx::new(db, range, true);
    let mut a = assists::all()
        .iter()
        .filter_map(|f| f(ctx.clone()))
        .map(|a| match a {
            Assist::Resolved { assist } => assist,
            Assist::Unresolved { .. } => unreachable!(),
        })
        .collect::<Vec<_>>();
    sort_assists(&mut a);
    a
}

fn sort_assists(assists: &mut [ResolvedAssist]) {
    assists.sort_by(|a, b| match (a.get_first_action().target, b.get_first_action().target) {
        (Some(a), Some(b)) => a.len().cmp(&b.len()),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
}

mod assists {
    use crate::{Assist, AssistCtx};

    mod add_derive;
    mod add_explicit_type;
    mod add_impl;
    mod add_custom_impl;
    mod add_new;
    mod apply_demorgan;
    mod auto_import;
    mod invert_if;
    mod flip_comma;
    mod flip_binexpr;
    mod flip_trait_bound;
    mod change_visibility;
    mod fill_match_arms;
    mod merge_match_arms;
    mod introduce_variable;
    mod inline_local_variable;
    mod raw_string;
    mod replace_if_let_with_match;
    mod split_import;
    mod remove_dbg;
    pub(crate) mod add_import;
    mod add_missing_impl_members;
    mod move_guard;
    mod move_bounds;
    mod early_return;

    pub(crate) fn all() -> &'static [fn(AssistCtx) -> Option<Assist>] {
        &[
            add_derive::add_derive,
            add_explicit_type::add_explicit_type,
            add_impl::add_impl,
            add_custom_impl::add_custom_impl,
            add_new::add_new,
            apply_demorgan::apply_demorgan,
            invert_if::invert_if,
            change_visibility::change_visibility,
            fill_match_arms::fill_match_arms,
            merge_match_arms::merge_match_arms,
            flip_comma::flip_comma,
            flip_binexpr::flip_binexpr,
            flip_trait_bound::flip_trait_bound,
            introduce_variable::introduce_variable,
            replace_if_let_with_match::replace_if_let_with_match,
            split_import::split_import,
            remove_dbg::remove_dbg,
            add_import::add_import,
            add_missing_impl_members::add_missing_impl_members,
            add_missing_impl_members::add_missing_default_members,
            inline_local_variable::inline_local_variable,
            move_guard::move_guard_to_arm_body,
            move_guard::move_arm_cond_to_match_guard,
            move_bounds::move_bounds_to_where_clause,
            raw_string::add_hash,
            raw_string::make_raw_string,
            raw_string::make_usual_string,
            raw_string::remove_hash,
            early_return::convert_to_guarded_return,
            auto_import::auto_import,
        ]
    }
}

#[cfg(test)]
mod helpers {
    use std::sync::Arc;

    use ra_db::{fixture::WithFixture, FileId, FileRange, SourceDatabaseExt};
    use ra_ide_db::{symbol_index::SymbolsDatabase, RootDatabase};
    use ra_syntax::TextRange;
    use test_utils::{add_cursor, assert_eq_text, extract_offset, extract_range};

    use crate::{Assist, AssistCtx};

    pub(crate) fn with_single_file(text: &str) -> (RootDatabase, FileId) {
        let (mut db, file_id) = RootDatabase::with_single_file(text);
        // FIXME: ideally, this should be done by the above `RootDatabase::with_single_file`,
        // but it looks like this might need specialization? :(
        let local_roots = vec![db.file_source_root(file_id)];
        db.set_local_roots(Arc::new(local_roots));
        (db, file_id)
    }

    pub(crate) fn check_assist(assist: fn(AssistCtx) -> Option<Assist>, before: &str, after: &str) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, file_id) = with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist =
            assist(AssistCtx::new(&db, frange, true)).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved { .. } => unreachable!(),
            Assist::Resolved { assist } => assist.get_first_action(),
        };

        let actual = action.edit.apply(&before);
        let actual_cursor_pos = match action.cursor_position {
            None => action
                .edit
                .apply_to_offset(before_cursor_pos)
                .expect("cursor position is affected by the edit"),
            Some(off) => off,
        };
        let actual = add_cursor(&actual, actual_cursor_pos);
        assert_eq_text!(after, &actual);
    }

    pub(crate) fn check_assist_range(
        assist: fn(AssistCtx) -> Option<Assist>,
        before: &str,
        after: &str,
    ) {
        let (range, before) = extract_range(before);
        let (db, file_id) = with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assist =
            assist(AssistCtx::new(&db, frange, true)).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved { .. } => unreachable!(),
            Assist::Resolved { assist } => assist.get_first_action(),
        };

        let mut actual = action.edit.apply(&before);
        if let Some(pos) = action.cursor_position {
            actual = add_cursor(&actual, pos);
        }
        assert_eq_text!(after, &actual);
    }

    pub(crate) fn check_assist_target(
        assist: fn(AssistCtx) -> Option<Assist>,
        before: &str,
        target: &str,
    ) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, file_id) = with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist =
            assist(AssistCtx::new(&db, frange, true)).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved { .. } => unreachable!(),
            Assist::Resolved { assist } => assist.get_first_action(),
        };

        let range = action.target.expect("expected target on action");
        assert_eq_text!(&before[range.start().to_usize()..range.end().to_usize()], target);
    }

    pub(crate) fn check_assist_range_target(
        assist: fn(AssistCtx) -> Option<Assist>,
        before: &str,
        target: &str,
    ) {
        let (range, before) = extract_range(before);
        let (db, file_id) = with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assist =
            assist(AssistCtx::new(&db, frange, true)).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved { .. } => unreachable!(),
            Assist::Resolved { assist } => assist.get_first_action(),
        };

        let range = action.target.expect("expected target on action");
        assert_eq_text!(&before[range.start().to_usize()..range.end().to_usize()], target);
    }

    pub(crate) fn check_assist_not_applicable(
        assist: fn(AssistCtx) -> Option<Assist>,
        before: &str,
    ) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, file_id) = with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist = assist(AssistCtx::new(&db, frange, true));
        assert!(assist.is_none());
    }

    pub(crate) fn check_assist_range_not_applicable(
        assist: fn(AssistCtx) -> Option<Assist>,
        before: &str,
    ) {
        let (range, before) = extract_range(before);
        let (db, file_id) = with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assist = assist(AssistCtx::new(&db, frange, true));
        assert!(assist.is_none());
    }
}

#[cfg(test)]
mod tests {
    use ra_db::FileRange;
    use ra_syntax::TextRange;
    use test_utils::{extract_offset, extract_range};

    use crate::{helpers, resolved_assists};

    #[test]
    fn assist_order_field_struct() {
        let before = "struct Foo { <|>bar: u32 }";
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, file_id) = helpers::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assists = resolved_assists(&db, frange);
        let mut assists = assists.iter();

        assert_eq!(
            assists.next().expect("expected assist").label.label,
            "Change visibility to pub(crate)"
        );
        assert_eq!(assists.next().expect("expected assist").label.label, "Add `#[derive]`");
    }

    #[test]
    fn assist_order_if_expr() {
        let before = "
        pub fn test_some_range(a: int) -> bool {
            if let 2..6 = <|>5<|> {
                true
            } else {
                false
            }
        }";
        let (range, before) = extract_range(before);
        let (db, file_id) = helpers::with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assists = resolved_assists(&db, frange);
        let mut assists = assists.iter();

        assert_eq!(assists.next().expect("expected assist").label.label, "Extract into variable");
        assert_eq!(assists.next().expect("expected assist").label.label, "Replace with match");
    }
}
