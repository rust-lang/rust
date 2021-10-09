//! This module contains tests for macro expansion. Effectively, it covers `tt`,
//! `mbe`, `proc_macro_api` and `hir_expand` crates. This might seem like a
//! wrong architecture at the first glance, but is intentional.
//!
//! Physically, macro expansion process is intertwined with name resolution. You
//! can not expand *just* the syntax. So, to be able to write integration tests
//! of the "expand this code please" form, we have to do it after name
//! resolution. That is, in this crate. We *could* fake some dependencies and
//! write unit-tests (in fact, we used to do that), but that makes tests brittle
//! and harder to understand.

use std::{iter, ops::Range};

use base_db::{fixture::WithFixture, SourceDatabase};
use expect_test::{expect, Expect};
use hir_expand::{db::AstDatabase, InFile, MacroFile};
use stdx::format_to;
use syntax::{
    ast, AstNode,
    SyntaxKind::{self, IDENT},
    SyntaxNode, T,
};

use crate::{
    db::DefDatabase, nameres::ModuleSource, resolver::HasResolver, test_db::TestDB, AsMacroCall,
};

fn check(ra_fixture: &str, expect: Expect) {
    let db = TestDB::with_files(ra_fixture);
    let krate = db.crate_graph().iter().next().unwrap();
    let def_map = db.crate_def_map(krate);
    let local_id = def_map.root();
    let module = def_map.module_id(local_id);
    let resolver = module.resolver(&db);
    let source = def_map[local_id].definition_source(&db);
    let source_file = match source.value {
        ModuleSource::SourceFile(it) => it,
        ModuleSource::Module(_) | ModuleSource::BlockExpr(_) => panic!(),
    };

    let mut expansions = Vec::new();
    for macro_call in source_file.syntax().descendants().filter_map(ast::MacroCall::cast) {
        let macro_call = InFile::new(source.file_id, &macro_call);
        let macro_call_id = macro_call
            .as_call_id_with_errors(
                &db,
                krate,
                |path| resolver.resolve_path_as_macro(&db, &path),
                &mut |err| panic!("{}", err),
            )
            .unwrap()
            .unwrap();
        let macro_file = MacroFile { macro_call_id };
        let expansion_result = db.parse_macro_expansion(macro_file);
        expansions.push((macro_call.value.clone(), expansion_result));
    }

    let mut expanded_text = source_file.to_string();
    for (call, exp) in expansions.into_iter().rev() {
        let mut expn_text = String::new();
        if let Some(err) = exp.err {
            format_to!(expn_text, "/* error: {} */", err);
        }
        if let Some((parse, _token_map)) = exp.value {
            let pp = pretty_print_macro_expansion(parse.syntax_node());
            format_to!(expn_text, "{}", pp);
        }
        let range = call.syntax().text_range();
        let range: Range<usize> = range.into();
        expanded_text.replace_range(range, &expn_text)
    }

    expect.assert_eq(&expanded_text);
}

fn pretty_print_macro_expansion(expn: SyntaxNode) -> String {
    let mut res = String::new();
    let mut prev_kind = SyntaxKind::EOF;
    for token in iter::successors(expn.first_token(), |t| t.next_token()) {
        let curr_kind = token.kind();
        let needs_space = match (prev_kind, curr_kind) {
            _ if prev_kind.is_trivia() || curr_kind.is_trivia() => false,
            (T![=], _) | (_, T![=]) => true,
            (IDENT, IDENT) => true,
            (IDENT, _) => curr_kind.is_keyword(),
            (_, IDENT) => prev_kind.is_keyword(),
            _ => false,
        };

        if needs_space {
            res.push(' ')
        }
        prev_kind = curr_kind;
        format_to!(res, "{}", token)
    }
    res
}

#[test]
fn wrong_nesting_level() {
    check(
        r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
m!{a}
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
/* error: expected simple binding, found nested binding `i` */
"#]],
    );
}

#[test]
fn expansion_does_not_parse_as_expression() {
    check(
        r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = stmts!(); }
"#,
        expect![[r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = /* error: could not convert tokens */; }
"#]],
    )
}

#[test]
fn round_trips_compound_tokens() {
    check(
        r#"
macro_rules! m {
    () => { type qual: ::T = qual::T; }
}
m!();
"#,
        expect![[r#"
macro_rules! m {
    () => { type qual: ::T = qual::T; }
}
type qual: ::T = qual::T;
        "#]],
    )
}
