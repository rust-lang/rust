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

use std::ops::Range;

use base_db::{fixture::WithFixture, SourceDatabase};
use expect_test::{expect, Expect};
use hir_expand::{db::AstDatabase, InFile, MacroFile};
use stdx::format_to;
use syntax::{ast, AstNode};

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
            format_to!(expn_text, "{}", parse.syntax_node());
        }
        let range = call.syntax().text_range();
        let range: Range<usize> = range.into();
        expanded_text.replace_range(range, &expn_text)
    }

    expect.assert_eq(&expanded_text);
}

#[test]
fn test_expand_rule() {
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
