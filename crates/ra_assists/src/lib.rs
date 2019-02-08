//! `ra_assits` crate provides a bunch of code assists, aslo known as code
//! actions (in LSP) or intentions (in IntelliJ).
//!
//! An assist is a micro-refactoring, which is automatically activated in
//! certain context. For example, if the cursor is over `,`, a "swap `,`" assist
//! becomes available.

mod assist_ctx;

use ra_text_edit::TextEdit;
use ra_syntax::{TextRange, TextUnit, SyntaxNode, Direction};
use ra_db::FileRange;
use hir::db::HirDatabase;

pub(crate) use crate::assist_ctx::{AssistCtx, Assist};

#[derive(Debug)]
pub struct AssistLabel {
    /// Short description of the assist, as shown in the UI.
    pub label: String,
}

pub struct AssistAction {
    pub edit: TextEdit,
    pub cursor_position: Option<TextUnit>,
    pub target: Option<TextRange>,
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "unresolved" state, that is only labels are
/// returned, without actual edits.
pub fn applicable_assists<H>(db: &H, range: FileRange) -> Vec<AssistLabel>
where
    H: HirDatabase + 'static,
{
    AssistCtx::with_ctx(db, range, false, |ctx| {
        all_assists()
            .iter()
            .filter_map(|f| f(ctx.clone()))
            .map(|a| match a {
                Assist::Unresolved(label) => label,
                Assist::Resolved(..) => unreachable!(),
            })
            .collect()
    })
}

/// Return all the assists applicable at the given position.
///
/// Assists are returned in the "resolved" state, that is with edit fully
/// computed.
pub fn assists<H>(db: &H, range: FileRange) -> Vec<(AssistLabel, AssistAction)>
where
    H: HirDatabase + 'static,
{
    use std::cmp::Ordering;

    AssistCtx::with_ctx(db, range, true, |ctx| {
        let mut a = all_assists()
            .iter()
            .filter_map(|f| f(ctx.clone()))
            .map(|a| match a {
                Assist::Resolved(label, action) => (label, action),
                Assist::Unresolved(..) => unreachable!(),
            })
            .collect::<Vec<(AssistLabel, AssistAction)>>();
        a.sort_by(|a, b| match a {
            // Some(y) < Some(x) < None for y < x
            (_, AssistAction { target: Some(a), .. }) => match b {
                (_, AssistAction { target: Some(b), .. }) => a.len().cmp(&b.len()),
                _ => Ordering::Less,
            },
            _ => Ordering::Greater,
        });
        a
    })
}

mod add_derive;
mod add_impl;
mod flip_comma;
mod change_visibility;
mod fill_match_arms;
mod introduce_variable;
mod replace_if_let_with_match;
mod split_import;
mod remove_dbg;
fn all_assists<DB: HirDatabase>() -> &'static [fn(AssistCtx<DB>) -> Option<Assist>] {
    &[
        add_derive::add_derive,
        add_impl::add_impl,
        change_visibility::change_visibility,
        fill_match_arms::fill_match_arms,
        flip_comma::flip_comma,
        introduce_variable::introduce_variable,
        replace_if_let_with_match::replace_if_let_with_match,
        split_import::split_import,
        remove_dbg::remove_dbg,
    ]
}

fn non_trivia_sibling(node: &SyntaxNode, direction: Direction) -> Option<&SyntaxNode> {
    node.siblings(direction).skip(1).find(|node| !node.kind().is_trivia())
}

#[cfg(test)]
mod helpers {
    use hir::mock::MockDatabase;
    use ra_syntax::TextRange;
    use ra_db::FileRange;
    use test_utils::{extract_offset, extract_range, assert_eq_text, add_cursor};

    use crate::{AssistCtx, Assist};

    pub(crate) fn check_assist(
        assist: fn(AssistCtx<MockDatabase>) -> Option<Assist>,
        before: &str,
        after: &str,
    ) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist =
            AssistCtx::with_ctx(&db, frange, true, assist).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved(_) => unreachable!(),
            Assist::Resolved(_, it) => it,
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
        assist: fn(AssistCtx<MockDatabase>) -> Option<Assist>,
        before: &str,
        after: &str,
    ) {
        let (range, before) = extract_range(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assist =
            AssistCtx::with_ctx(&db, frange, true, assist).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved(_) => unreachable!(),
            Assist::Resolved(_, it) => it,
        };

        let mut actual = action.edit.apply(&before);
        if let Some(pos) = action.cursor_position {
            actual = add_cursor(&actual, pos);
        }
        assert_eq_text!(after, &actual);
    }

    pub(crate) fn check_assist_target(
        assist: fn(AssistCtx<MockDatabase>) -> Option<Assist>,
        before: &str,
        target: &str,
    ) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist =
            AssistCtx::with_ctx(&db, frange, true, assist).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved(_) => unreachable!(),
            Assist::Resolved(_, it) => it,
        };

        let range = action.target.expect("expected target on action");
        assert_eq_text!(&before[range.start().to_usize()..range.end().to_usize()], target);
    }

    pub(crate) fn check_assist_range_target(
        assist: fn(AssistCtx<MockDatabase>) -> Option<Assist>,
        before: &str,
        target: &str,
    ) {
        let (range, before) = extract_range(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange = FileRange { file_id, range };
        let assist =
            AssistCtx::with_ctx(&db, frange, true, assist).expect("code action is not applicable");
        let action = match assist {
            Assist::Unresolved(_) => unreachable!(),
            Assist::Resolved(_, it) => it,
        };

        let range = action.target.expect("expected target on action");
        assert_eq_text!(&before[range.start().to_usize()..range.end().to_usize()], target);
    }

    pub(crate) fn check_assist_not_applicable(
        assist: fn(AssistCtx<MockDatabase>) -> Option<Assist>,
        before: &str,
    ) {
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assist = AssistCtx::with_ctx(&db, frange, true, assist);
        assert!(assist.is_none());
    }
}

#[cfg(test)]
mod tests {
    use hir::mock::MockDatabase;
    use ra_syntax::TextRange;
    use ra_db::FileRange;
    use test_utils::{extract_offset};

    #[test]
    fn assist_order_field_struct() {
        let before = "struct Foo { <|>bar: u32 }";
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assists = super::assists(&db, frange);
        let mut assists = assists.iter();

        assert_eq!(assists.next().expect("expected assist").0.label, "make pub(crate)");
        assert_eq!(assists.next().expect("expected assist").0.label, "add `#[derive]`");
    }

    #[test]
    fn assist_order_if_expr() {
        let before = "
        pub fn test_some_range(a: int) -> bool {
            if let 2..6 = 5<|> {
                true
            } else {
                false
            }
        }";
        let (before_cursor_pos, before) = extract_offset(before);
        let (db, _source_root, file_id) = MockDatabase::with_single_file(&before);
        let frange =
            FileRange { file_id, range: TextRange::offset_len(before_cursor_pos, 0.into()) };
        let assists = super::assists(&db, frange);
        let mut assists = assists.iter();

        assert_eq!(assists.next().expect("expected assist").0.label, "introduce variable");
        assert_eq!(assists.next().expect("expected assist").0.label, "replace with match");
    }

}
