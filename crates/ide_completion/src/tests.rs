//! Tests and test utilities for completions.
//!
//! Most tests live in this module or its submodules. The tests in these submodules are "location"
//! oriented, that is they try to check completions for something like type position, param position
//! etc.
//! Tests that are more orientated towards specific completion types like visibility checks of path
//! completions or `check_edit` tests usually live in their respective completion modules instead.
//! This gives this test module and its submodules here the main purpose of giving the developer an
//! overview of whats being completed where, not how.

mod attribute;
mod expression;
mod fn_param;
mod item_list;
mod item;
mod pattern;
mod predicate;
mod proc_macros;
mod record;
mod type_pos;
mod use_tree;
mod visibility;
mod flyimport;

use std::mem;

use hir::{db::DefDatabase, PrefixKind, Semantics};
use ide_db::{
    base_db::{fixture::ChangeFixture, FileLoader, FilePosition},
    helpers::{
        insert_use::{ImportGranularity, InsertUseConfig},
        SnippetCap,
    },
    RootDatabase,
};
use itertools::Itertools;
use stdx::{format_to, trim_indent};
use syntax::{AstNode, NodeOrToken, SyntaxElement};
use test_utils::assert_eq_text;

use crate::{CompletionConfig, CompletionItem, CompletionItemKind};

/// Lots of basic item definitions
const BASE_ITEMS_FIXTURE: &str = r#"
enum Enum { TupleV(u32), RecordV { field: u32 }, UnitV }
use self::Enum::TupleV;
mod module {}

trait Trait {}
static STATIC: Unit = Unit;
const CONST: Unit = Unit;
struct Record { field: u32 }
struct Tuple(u32);
struct Unit;
#[macro_export]
macro_rules! makro {}
#[rustc_builtin_macro]
pub macro Clone {}
fn function() {}
union Union { field: i32 }
"#;

pub(crate) const TEST_CONFIG: CompletionConfig = CompletionConfig {
    enable_postfix_completions: true,
    enable_imports_on_the_fly: true,
    enable_self_on_the_fly: true,
    add_call_parenthesis: true,
    add_call_argument_snippets: true,
    snippet_cap: SnippetCap::new(true),
    insert_use: InsertUseConfig {
        granularity: ImportGranularity::Crate,
        prefix_kind: PrefixKind::Plain,
        enforce_granularity: true,
        group: true,
        skip_glob_imports: true,
    },
    snippets: Vec::new(),
};

pub(crate) fn completion_list(ra_fixture: &str) -> String {
    completion_list_with_config(TEST_CONFIG, ra_fixture, true)
}

pub(crate) fn completion_list_no_kw(ra_fixture: &str) -> String {
    completion_list_with_config(TEST_CONFIG, ra_fixture, false)
}

fn completion_list_with_config(
    config: CompletionConfig,
    ra_fixture: &str,
    include_keywords: bool,
) -> String {
    // filter out all but one builtintype completion for smaller test outputs
    let items = get_all_items(config, ra_fixture);
    let mut bt_seen = false;
    let items = items
        .into_iter()
        .filter(|it| {
            it.kind() != CompletionItemKind::BuiltinType || !mem::replace(&mut bt_seen, true)
        })
        .filter(|it| include_keywords || it.kind() != CompletionItemKind::Keyword)
        .filter(|it| include_keywords || it.kind() != CompletionItemKind::Snippet)
        .collect();
    render_completion_list(items)
}

/// Creates analysis from a multi-file fixture, returns positions marked with $0.
pub(crate) fn position(ra_fixture: &str) -> (RootDatabase, FilePosition) {
    let change_fixture = ChangeFixture::parse(ra_fixture);
    let mut database = RootDatabase::default();
    database.set_enable_proc_attr_macros(true);
    database.apply_change(change_fixture.change);
    let (file_id, range_or_offset) = change_fixture.file_position.expect("expected a marker ($0)");
    let offset = range_or_offset.expect_offset();
    (database, FilePosition { file_id, offset })
}

pub(crate) fn do_completion(code: &str, kind: CompletionItemKind) -> Vec<CompletionItem> {
    do_completion_with_config(TEST_CONFIG, code, kind)
}

pub(crate) fn do_completion_with_config(
    config: CompletionConfig,
    code: &str,
    kind: CompletionItemKind,
) -> Vec<CompletionItem> {
    get_all_items(config, code)
        .into_iter()
        .filter(|c| c.kind() == kind)
        .sorted_by(|l, r| l.label().cmp(r.label()))
        .collect()
}

fn render_completion_list(completions: Vec<CompletionItem>) -> String {
    fn monospace_width(s: &str) -> usize {
        s.chars().count()
    }
    let label_width =
        completions.iter().map(|it| monospace_width(it.label())).max().unwrap_or_default().min(22);
    completions
        .into_iter()
        .map(|it| {
            let tag = it.kind().tag();
            let var_name = format!("{} {}", tag, it.label());
            let mut buf = var_name;
            if let Some(detail) = it.detail() {
                let width = label_width.saturating_sub(monospace_width(it.label()));
                format_to!(buf, "{:width$} {}", "", detail, width = width);
            }
            if it.deprecated() {
                format_to!(buf, " DEPRECATED");
            }
            format_to!(buf, "\n");
            buf
        })
        .collect()
}

pub(crate) fn check_edit(what: &str, ra_fixture_before: &str, ra_fixture_after: &str) {
    check_edit_with_config(TEST_CONFIG, what, ra_fixture_before, ra_fixture_after)
}

pub(crate) fn check_edit_with_config(
    config: CompletionConfig,
    what: &str,
    ra_fixture_before: &str,
    ra_fixture_after: &str,
) {
    let ra_fixture_after = trim_indent(ra_fixture_after);
    let (db, position) = position(ra_fixture_before);
    let completions: Vec<CompletionItem> =
        crate::completions(&db, &config, position).unwrap().into();
    let (completion,) = completions
        .iter()
        .filter(|it| it.lookup() == what)
        .collect_tuple()
        .unwrap_or_else(|| panic!("can't find {:?} completion in {:#?}", what, completions));
    let mut actual = db.file_text(position.file_id).to_string();

    let mut combined_edit = completion.text_edit().to_owned();
    completion
        .imports_to_add()
        .iter()
        .filter_map(|edit| edit.to_text_edit(config.insert_use))
        .for_each(|text_edit| {
            combined_edit.union(text_edit).expect(
                "Failed to apply completion resolve changes: change ranges overlap, but should not",
            )
        });

    combined_edit.apply(&mut actual);
    assert_eq_text!(&ra_fixture_after, &actual)
}

pub(crate) fn check_pattern_is_applicable(code: &str, check: impl FnOnce(SyntaxElement) -> bool) {
    let (db, pos) = position(code);

    let sema = Semantics::new(&db);
    let original_file = sema.parse(pos.file_id);
    let token = original_file.syntax().token_at_offset(pos.offset).left_biased().unwrap();
    assert!(check(NodeOrToken::Token(token)));
}

pub(crate) fn check_pattern_is_not_applicable(code: &str, check: fn(SyntaxElement) -> bool) {
    let (db, pos) = position(code);
    let sema = Semantics::new(&db);
    let original_file = sema.parse(pos.file_id);
    let token = original_file.syntax().token_at_offset(pos.offset).left_biased().unwrap();
    assert!(!check(NodeOrToken::Token(token)));
}

pub(crate) fn get_all_items(config: CompletionConfig, code: &str) -> Vec<CompletionItem> {
    let (db, position) = position(code);
    crate::completions(&db, &config, position).map_or_else(Vec::default, Into::into)
}

fn check_no_completion(ra_fixture: &str) {
    let (db, position) = position(ra_fixture);

    assert!(
        crate::completions(&db, &TEST_CONFIG, position).is_none(),
        "Completions were generated, but weren't expected"
    );
}

#[test]
fn test_no_completions_required() {
    cov_mark::check!(no_completion_required);
    check_no_completion(r#"fn foo() { for i i$0 }"#);
}

#[test]
fn regression_10042() {
    completion_list(
        r#"
macro_rules! preset {
    ($($x:ident)&&*) => {
        {
            let mut v = Vec::new();
            $(
                v.push($x.into());
            )*
            v
        }
    };
}

fn foo() {
    preset!(foo$0);
}
"#,
    );
}

#[test]
fn no_completions_in_comments() {
    cov_mark::check!(no_keyword_completion_in_comments);
    assert_eq!(
        completion_list(
            r#"
fn test() {
let x = 2; // A comment$0
}
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/*
Some multi-line comment$0
*/
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/// Some doc comment
/// let test$0 = 1
"#,
        ),
        String::new(),
    );
}
