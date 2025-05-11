use expect_test::{Expect, expect};
use hir::{FilePosition, FileRange};
use ide_db::{
    EditionedFileId, FxHashSet,
    base_db::{SourceDatabase, salsa::Durability},
};
use test_utils::RangeOrOffset;
use triomphe::Arc;

use crate::{MatchFinder, SsrRule};

fn parse_error_text(query: &str) -> String {
    format!("{}", query.parse::<SsrRule>().unwrap_err())
}

#[test]
fn parser_empty_query() {
    assert_eq!(parse_error_text(""), "Parse error: Cannot find delimiter `==>>`");
}

#[test]
fn parser_no_delimiter() {
    assert_eq!(parse_error_text("foo()"), "Parse error: Cannot find delimiter `==>>`");
}

#[test]
fn parser_two_delimiters() {
    assert_eq!(
        parse_error_text("foo() ==>> a ==>> b "),
        "Parse error: More than one delimiter found"
    );
}

#[test]
fn parser_repeated_name() {
    assert_eq!(
        parse_error_text("foo($a, $a) ==>>"),
        "Parse error: Placeholder `$a` repeats more than once"
    );
}

#[test]
fn parser_invalid_pattern() {
    assert_eq!(
        parse_error_text(" ==>> ()"),
        "Parse error: Not a valid Rust expression, type, item, path or pattern"
    );
}

#[test]
fn parser_invalid_template() {
    assert_eq!(
        parse_error_text("() ==>> )"),
        "Parse error: Not a valid Rust expression, type, item, path or pattern"
    );
}

#[test]
fn parser_undefined_placeholder_in_replacement() {
    assert_eq!(
        parse_error_text("42 ==>> $a"),
        "Parse error: Replacement contains undefined placeholders: $a"
    );
}

/// `code` may optionally contain a cursor marker `$0`. If it doesn't, then the position will be
/// the start of the file. If there's a second cursor marker, then we'll return a single range.
pub(crate) fn single_file(code: &str) -> (ide_db::RootDatabase, FilePosition, Vec<FileRange>) {
    use ide_db::symbol_index::SymbolsDatabase;
    use test_fixture::{WORKSPACE, WithFixture};
    let (mut db, file_id, range_or_offset) = if code.contains(test_utils::CURSOR_MARKER) {
        ide_db::RootDatabase::with_range_or_offset(code)
    } else {
        let (db, file_id) = ide_db::RootDatabase::with_single_file(code);
        (db, file_id, RangeOrOffset::Offset(0.into()))
    };
    let selections;
    let position;
    match range_or_offset {
        RangeOrOffset::Range(range) => {
            position = FilePosition { file_id, offset: range.start() };
            selections = vec![FileRange { file_id, range }];
        }
        RangeOrOffset::Offset(offset) => {
            position = FilePosition { file_id, offset };
            selections = vec![];
        }
    }
    let mut local_roots = FxHashSet::default();
    local_roots.insert(WORKSPACE);
    db.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);
    (db, position, selections)
}

fn assert_ssr_transform(rule: &str, input: &str, expected: Expect) {
    assert_ssr_transforms(&[rule], input, expected);
}

fn assert_ssr_transforms(rules: &[&str], input: &str, expected: Expect) {
    let (db, position, selections) = single_file(input);
    let position =
        ide_db::FilePosition { file_id: position.file_id.file_id(&db), offset: position.offset };
    let mut match_finder = MatchFinder::in_context(
        &db,
        position,
        selections
            .into_iter()
            .map(|selection| ide_db::FileRange {
                file_id: selection.file_id.file_id(&db),
                range: selection.range,
            })
            .collect(),
    )
    .unwrap();
    for rule in rules {
        let rule: SsrRule = rule.parse().unwrap();
        match_finder.add_rule(rule).unwrap();
    }
    let edits = match_finder.edits();
    if edits.is_empty() {
        panic!("No edits were made");
    }
    // Note, db.file_text is not necessarily the same as `input`, since fixture parsing alters
    // stuff.
    let mut actual = db.file_text(position.file_id).text(&db).to_string();
    edits[&position.file_id].apply(&mut actual);
    expected.assert_eq(&actual);
}

#[allow(clippy::print_stdout)]
fn print_match_debug_info(match_finder: &MatchFinder<'_>, file_id: EditionedFileId, snippet: &str) {
    let debug_info = match_finder.debug_where_text_equal(file_id, snippet);
    println!(
        "Match debug info: {} nodes had text exactly equal to '{}'",
        debug_info.len(),
        snippet
    );
    for (index, d) in debug_info.iter().enumerate() {
        println!("Node #{index}\n{d:#?}\n");
    }
}

fn assert_matches(pattern: &str, code: &str, expected: &[&str]) {
    let (db, position, selections) = single_file(code);
    let mut match_finder = MatchFinder::in_context(
        &db,
        ide_db::FilePosition { file_id: position.file_id.file_id(&db), offset: position.offset },
        selections
            .into_iter()
            .map(|selection| ide_db::FileRange {
                file_id: selection.file_id.file_id(&db),
                range: selection.range,
            })
            .collect(),
    )
    .unwrap();
    match_finder.add_search_pattern(pattern.parse().unwrap()).unwrap();
    let matched_strings: Vec<String> =
        match_finder.matches().flattened().matches.iter().map(|m| m.matched_text()).collect();
    if matched_strings != expected && !expected.is_empty() {
        print_match_debug_info(&match_finder, position.file_id, expected[0]);
    }
    assert_eq!(matched_strings, expected);
}

fn assert_no_match(pattern: &str, code: &str) {
    let (db, position, selections) = single_file(code);
    let mut match_finder = MatchFinder::in_context(
        &db,
        ide_db::FilePosition { file_id: position.file_id.file_id(&db), offset: position.offset },
        selections
            .into_iter()
            .map(|selection| ide_db::FileRange {
                file_id: selection.file_id.file_id(&db),
                range: selection.range,
            })
            .collect(),
    )
    .unwrap();
    match_finder.add_search_pattern(pattern.parse().unwrap()).unwrap();
    let matches = match_finder.matches().flattened().matches;
    if !matches.is_empty() {
        print_match_debug_info(&match_finder, position.file_id, &matches[0].matched_text());
        panic!("Got {} matches when we expected none: {matches:#?}", matches.len());
    }
}

fn assert_match_failure_reason(pattern: &str, code: &str, snippet: &str, expected_reason: &str) {
    let (db, position, selections) = single_file(code);
    let mut match_finder = MatchFinder::in_context(
        &db,
        ide_db::FilePosition { file_id: position.file_id.file_id(&db), offset: position.offset },
        selections
            .into_iter()
            .map(|selection| ide_db::FileRange {
                file_id: selection.file_id.file_id(&db),
                range: selection.range,
            })
            .collect(),
    )
    .unwrap();
    match_finder.add_search_pattern(pattern.parse().unwrap()).unwrap();
    let mut reasons = Vec::new();
    for d in match_finder.debug_where_text_equal(position.file_id, snippet) {
        if let Some(reason) = d.match_failure_reason() {
            reasons.push(reason.to_owned());
        }
    }
    assert_eq!(reasons, vec![expected_reason]);
}

#[test]
fn ssr_let_stmt_in_macro_match() {
    assert_matches(
        "let a = 0",
        r#"
            macro_rules! m1 { ($a:stmt) => {$a}; }
            fn f() {m1!{ let a = 0 };}"#,
        // FIXME: Whitespace is not part of the matched block
        &["leta=0"],
    );
}

#[test]
fn ssr_let_stmt_in_fn_match() {
    assert_matches("let $a = 10;", "fn main() { let x = 10; x }", &["let x = 10;"]);
    assert_matches("let $a = $b;", "fn main() { let x = 10; x }", &["let x = 10;"]);
}

#[test]
fn ssr_block_expr_match() {
    assert_matches("{ let $a = $b; }", "fn main() { let x = 10; }", &["{ let x = 10; }"]);
    assert_matches("{ let $a = $b; $c }", "fn main() { let x = 10; x }", &["{ let x = 10; x }"]);
}

#[test]
fn ssr_let_stmt_replace() {
    // Pattern and template with trailing semicolon
    assert_ssr_transform(
        "let $a = $b; ==>> let $a = 11;",
        "fn main() { let x = 10; x }",
        expect![["fn main() { let x = 11; x }"]],
    );
}

#[test]
fn ssr_let_stmt_replace_expr() {
    // Trailing semicolon should be dropped from the new expression
    assert_ssr_transform(
        "let $a = $b; ==>> $b",
        "fn main() { let x = 10; }",
        expect![["fn main() { 10 }"]],
    );
}

#[test]
fn ssr_blockexpr_replace_stmt_with_stmt() {
    assert_ssr_transform(
        "if $a() {$b;} ==>> $b;",
        "{
    if foo() {
        bar();
    }
    Ok(())
}",
        expect![[r#"{
    bar();
    Ok(())
}"#]],
    );
}

#[test]
fn ssr_blockexpr_match_trailing_expr() {
    assert_matches(
        "if $a() {$b;}",
        "{
    if foo() {
        bar();
    }
}",
        &["if foo() {
        bar();
    }"],
    );
}

#[test]
fn ssr_blockexpr_replace_trailing_expr_with_stmt() {
    assert_ssr_transform(
        "if $a() {$b;} ==>> $b;",
        "{
    if foo() {
        bar();
    }
}",
        expect![["{
    bar();
}"]],
    );
}

#[test]
fn ssr_function_to_method() {
    assert_ssr_transform(
        "my_function($a, $b) ==>> ($a).my_method($b)",
        "fn my_function() {} fn main() { loop { my_function( other_func(x, y), z + w) } }",
        expect![["fn my_function() {} fn main() { loop { (other_func(x, y)).my_method(z + w) } }"]],
    )
}

#[test]
fn ssr_nested_function() {
    assert_ssr_transform(
        "foo($a, $b, $c) ==>> bar($c, baz($a, $b))",
        r#"
            //- /lib.rs crate:foo
            fn foo() {}
            fn bar() {}
            fn baz() {}
            fn main { foo  (x + value.method(b), x+y-z, true && false) }
            "#,
        expect![[r#"
            fn foo() {}
            fn bar() {}
            fn baz() {}
            fn main { bar(true && false, baz(x + value.method(b), x+y-z)) }
        "#]],
    )
}

#[test]
fn ssr_expected_spacing() {
    assert_ssr_transform(
        "foo($x) + bar() ==>> bar($x)",
        "fn foo() {} fn bar() {} fn main() { foo(5) + bar() }",
        expect![["fn foo() {} fn bar() {} fn main() { bar(5) }"]],
    );
}

#[test]
fn ssr_with_extra_space() {
    assert_ssr_transform(
        "foo($x  ) +    bar() ==>> bar($x)",
        "fn foo() {} fn bar() {} fn main() { foo(  5 )  +bar(   ) }",
        expect![["fn foo() {} fn bar() {} fn main() { bar(5) }"]],
    );
}

#[test]
fn ssr_keeps_nested_comment() {
    assert_ssr_transform(
        "foo($x) ==>> bar($x)",
        "fn foo() {} fn bar() {} fn main() { foo(other(5 /* using 5 */)) }",
        expect![["fn foo() {} fn bar() {} fn main() { bar(other(5 /* using 5 */)) }"]],
    )
}

#[test]
fn ssr_keeps_comment() {
    assert_ssr_transform(
        "foo($x) ==>> bar($x)",
        "fn foo() {} fn bar() {} fn main() { foo(5 /* using 5 */) }",
        expect![["fn foo() {} fn bar() {} fn main() { bar(5)/* using 5 */ }"]],
    )
}

#[test]
fn ssr_struct_lit() {
    assert_ssr_transform(
        "Foo{a: $a, b: $b} ==>> Foo::new($a, $b)",
        r#"
            struct Foo() {}
            impl Foo { fn new() {} }
            fn main() { Foo{b:2, a:1} }
            "#,
        expect![[r#"
            struct Foo() {}
            impl Foo { fn new() {} }
            fn main() { Foo::new(1, 2) }
        "#]],
    )
}

#[test]
fn ssr_struct_def() {
    assert_ssr_transform(
        "struct Foo { $f: $t } ==>> struct Foo($t);",
        r#"struct Foo { field: i32 }"#,
        expect![[r#"struct Foo(i32);"#]],
    )
}

#[test]
fn ignores_whitespace() {
    assert_matches("1+2", "fn f() -> i32 {1  +  2}", &["1  +  2"]);
    assert_matches("1 + 2", "fn f() -> i32 {1+2}", &["1+2"]);
}

#[test]
fn no_match() {
    assert_no_match("1 + 3", "fn f() -> i32 {1  +  2}");
}

#[test]
fn match_fn_definition() {
    assert_matches("fn $a($b: $t) {$c}", "fn f(a: i32) {bar()}", &["fn f(a: i32) {bar()}"]);
}

#[test]
fn match_struct_definition() {
    let code = r#"
        struct Option<T> {}
        struct Bar {}
        struct Foo {name: Option<String>}"#;
    assert_matches("struct $n {$f: Option<String>}", code, &["struct Foo {name: Option<String>}"]);
}

#[test]
fn match_expr() {
    let code = r#"
        fn foo() {}
        fn f() -> i32 {foo(40 + 2, 42)}"#;
    assert_matches("foo($a, $b)", code, &["foo(40 + 2, 42)"]);
    assert_no_match("foo($a, $b, $c)", code);
    assert_no_match("foo($a)", code);
}

#[test]
fn match_nested_method_calls() {
    assert_matches(
        "$a.z().z().z()",
        "fn f() {h().i().j().z().z().z().d().e()}",
        &["h().i().j().z().z().z()"],
    );
}

// Make sure that our node matching semantics don't differ within macro calls.
#[test]
fn match_nested_method_calls_with_macro_call() {
    assert_matches(
        "$a.z().z().z()",
        r#"
            macro_rules! m1 { ($a:expr) => {$a}; }
            fn f() {m1!(h().i().j().z().z().z().d().e())}"#,
        &["h().i().j().z().z().z()"],
    );
}

#[test]
fn match_complex_expr() {
    let code = r#"
        fn foo() {} fn bar() {}
        fn f() -> i32 {foo(bar(40, 2), 42)}"#;
    assert_matches("foo($a, $b)", code, &["foo(bar(40, 2), 42)"]);
    assert_no_match("foo($a, $b, $c)", code);
    assert_no_match("foo($a)", code);
    assert_matches("bar($a, $b)", code, &["bar(40, 2)"]);
}

// Trailing commas in the code should be ignored.
#[test]
fn match_with_trailing_commas() {
    // Code has comma, pattern doesn't.
    assert_matches("foo($a, $b)", "fn foo() {} fn f() {foo(1, 2,);}", &["foo(1, 2,)"]);
    assert_matches("Foo{$a, $b}", "struct Foo {} fn f() {Foo{1, 2,};}", &["Foo{1, 2,}"]);

    // Pattern has comma, code doesn't.
    assert_matches("foo($a, $b,)", "fn foo() {} fn f() {foo(1, 2);}", &["foo(1, 2)"]);
    assert_matches("Foo{$a, $b,}", "struct Foo {} fn f() {Foo{1, 2};}", &["Foo{1, 2}"]);
}

#[test]
fn match_type() {
    assert_matches("i32", "fn f() -> i32 {1  +  2}", &["i32"]);
    assert_matches(
        "Option<$a>",
        "struct Option<T> {} fn f() -> Option<i32> {42}",
        &["Option<i32>"],
    );
    assert_no_match(
        "Option<$a>",
        "struct Option<T> {} struct Result<T, E> {} fn f() -> Result<i32, ()> {42}",
    );
}

#[test]
fn match_struct_instantiation() {
    let code = r#"
        struct Foo {bar: i32, baz: i32}
        fn f() {Foo {bar: 1, baz: 2}}"#;
    assert_matches("Foo {bar: 1, baz: 2}", code, &["Foo {bar: 1, baz: 2}"]);
    // Now with placeholders for all parts of the struct.
    assert_matches("Foo {$a: $b, $c: $d}", code, &["Foo {bar: 1, baz: 2}"]);
    assert_matches("Foo {}", "struct Foo {} fn f() {Foo {}}", &["Foo {}"]);
}

#[test]
fn match_path() {
    let code = r#"
        mod foo {
            pub(crate) fn bar() {}
        }
        fn f() {foo::bar(42)}"#;
    assert_matches("foo::bar", code, &["foo::bar"]);
    assert_matches("$a::bar", code, &["foo::bar"]);
    assert_matches("foo::$b", code, &["foo::bar"]);
}

#[test]
fn match_pattern() {
    assert_matches("Some($a)", "struct Some(); fn f() {if let Some(x) = foo() {}}", &["Some(x)"]);
}

// If our pattern has a full path, e.g. a::b::c() and the code has c(), but c resolves to
// a::b::c, then we should match.
#[test]
fn match_fully_qualified_fn_path() {
    let code = r#"
        mod a {
            pub(crate) mod b {
                pub(crate) fn c(_: i32) {}
            }
        }
        use a::b::c;
        fn f1() {
            c(42);
        }
        "#;
    assert_matches("a::b::c($a)", code, &["c(42)"]);
}

#[test]
fn match_resolved_type_name() {
    let code = r#"
        mod m1 {
            pub(crate) mod m2 {
                pub(crate) trait Foo<T> {}
            }
        }
        mod m3 {
            trait Foo<T> {}
            fn f1(f: Option<&dyn Foo<bool>>) {}
        }
        mod m4 {
            use crate::m1::m2::Foo;
            fn f1(f: Option<&dyn Foo<i32>>) {}
        }
        "#;
    assert_matches("m1::m2::Foo<$t>", code, &["Foo<i32>"]);
}

#[test]
fn type_arguments_within_path() {
    cov_mark::check!(type_arguments_within_path);
    let code = r#"
        mod foo {
            pub(crate) struct Bar<T> {t: T}
            impl<T> Bar<T> {
                pub(crate) fn baz() {}
            }
        }
        fn f1() {foo::Bar::<i32>::baz();}
        "#;
    assert_no_match("foo::Bar::<i64>::baz()", code);
    assert_matches("foo::Bar::<i32>::baz()", code, &["foo::Bar::<i32>::baz()"]);
}

#[test]
fn literal_constraint() {
    cov_mark::check!(literal_constraint);
    let code = r#"
        enum Option<T> { Some(T), None }
        use Option::Some;
        fn f1() {
            let x1 = Some(42);
            let x2 = Some("foo");
            let x3 = Some(x1);
            let x4 = Some(40 + 2);
            let x5 = Some(true);
        }
        "#;
    assert_matches("Some(${a:kind(literal)})", code, &["Some(42)", "Some(\"foo\")", "Some(true)"]);
    assert_matches("Some(${a:not(kind(literal))})", code, &["Some(x1)", "Some(40 + 2)"]);
}

#[test]
fn match_reordered_struct_instantiation() {
    assert_matches(
        "Foo {aa: 1, b: 2, ccc: 3}",
        "struct Foo {} fn f() {Foo {b: 2, ccc: 3, aa: 1}}",
        &["Foo {b: 2, ccc: 3, aa: 1}"],
    );
    assert_no_match("Foo {a: 1}", "struct Foo {} fn f() {Foo {b: 1}}");
    assert_no_match("Foo {a: 1}", "struct Foo {} fn f() {Foo {a: 2}}");
    assert_no_match("Foo {a: 1, b: 2}", "struct Foo {} fn f() {Foo {a: 1}}");
    assert_no_match("Foo {a: 1, b: 2}", "struct Foo {} fn f() {Foo {b: 2}}");
    assert_no_match("Foo {a: 1, }", "struct Foo {} fn f() {Foo {a: 1, b: 2}}");
    assert_no_match("Foo {a: 1, z: 9}", "struct Foo {} fn f() {Foo {a: 1}}");
}

#[test]
fn match_macro_invocation() {
    assert_matches(
        "foo!($a)",
        "macro_rules! foo {() => {}} fn() {foo(foo!(foo()))}",
        &["foo!(foo())"],
    );
    assert_matches(
        "foo!(41, $a, 43)",
        "macro_rules! foo {() => {}} fn() {foo!(41, 42, 43)}",
        &["foo!(41, 42, 43)"],
    );
    assert_no_match("foo!(50, $a, 43)", "macro_rules! foo {() => {}} fn() {foo!(41, 42, 43}");
    assert_no_match("foo!(41, $a, 50)", "macro_rules! foo {() => {}} fn() {foo!(41, 42, 43}");
    assert_matches(
        "foo!($a())",
        "macro_rules! foo {() => {}} fn() {foo!(bar())}",
        &["foo!(bar())"],
    );
}

// When matching within a macro expansion, we only allow matches of nodes that originated from
// the macro call, not from the macro definition.
#[test]
fn no_match_expression_from_macro() {
    assert_no_match(
        "$a.clone()",
        r#"
            macro_rules! m1 {
                () => {42.clone()}
            }
            fn f1() {m1!()}
            "#,
    );
}

// We definitely don't want to allow matching of an expression that part originates from the
// macro call `42` and part from the macro definition `.clone()`.
#[test]
fn no_match_split_expression() {
    assert_no_match(
        "$a.clone()",
        r#"
            macro_rules! m1 {
                ($x:expr) => {$x.clone()}
            }
            fn f1() {m1!(42)}
            "#,
    );
}

#[test]
fn replace_function_call() {
    // This test also makes sure that we ignore empty-ranges.
    assert_ssr_transform(
        "foo() ==>> bar()",
        "fn foo() {$0$0} fn bar() {} fn f1() {foo(); foo();}",
        expect![["fn foo() {} fn bar() {} fn f1() {bar(); bar();}"]],
    );
}

#[test]
fn replace_function_call_with_placeholders() {
    assert_ssr_transform(
        "foo($a, $b) ==>> bar($b, $a)",
        "fn foo() {} fn bar() {} fn f1() {foo(5, 42)}",
        expect![["fn foo() {} fn bar() {} fn f1() {bar(42, 5)}"]],
    );
}

#[test]
fn replace_nested_function_calls() {
    assert_ssr_transform(
        "foo($a) ==>> bar($a)",
        "fn foo() {} fn bar() {} fn f1() {foo(foo(42))}",
        expect![["fn foo() {} fn bar() {} fn f1() {bar(bar(42))}"]],
    );
}

#[test]
fn replace_associated_function_call() {
    assert_ssr_transform(
        "Foo::new() ==>> Bar::new()",
        r#"
            struct Foo {}
            impl Foo { fn new() {} }
            struct Bar {}
            impl Bar { fn new() {} }
            fn f1() {Foo::new();}
            "#,
        expect![[r#"
            struct Foo {}
            impl Foo { fn new() {} }
            struct Bar {}
            impl Bar { fn new() {} }
            fn f1() {Bar::new();}
        "#]],
    );
}

#[test]
fn replace_associated_trait_default_function_call() {
    cov_mark::check!(replace_associated_trait_default_function_call);
    assert_ssr_transform(
        "Bar2::foo() ==>> Bar2::foo2()",
        r#"
            trait Foo { fn foo() {} }
            pub(crate) struct Bar {}
            impl Foo for Bar {}
            pub(crate) struct Bar2 {}
            impl Foo for Bar2 {}
            impl Bar2 { fn foo2() {} }
            fn main() {
                Bar::foo();
                Bar2::foo();
            }
        "#,
        expect![[r#"
            trait Foo { fn foo() {} }
            pub(crate) struct Bar {}
            impl Foo for Bar {}
            pub(crate) struct Bar2 {}
            impl Foo for Bar2 {}
            impl Bar2 { fn foo2() {} }
            fn main() {
                Bar::foo();
                Bar2::foo2();
            }
        "#]],
    );
}

#[test]
fn replace_associated_trait_constant() {
    cov_mark::check!(replace_associated_trait_constant);
    assert_ssr_transform(
        "Bar2::VALUE ==>> Bar2::VALUE_2222",
        r#"
            trait Foo { const VALUE: i32; const VALUE_2222: i32; }
            pub(crate) struct Bar {}
            impl Foo for Bar { const VALUE: i32 = 1;  const VALUE_2222: i32 = 2; }
            pub(crate) struct Bar2 {}
            impl Foo for Bar2 { const VALUE: i32 = 1;  const VALUE_2222: i32 = 2; }
            impl Bar2 { fn foo2() {} }
            fn main() {
                Bar::VALUE;
                Bar2::VALUE;
            }
            "#,
        expect![[r#"
            trait Foo { const VALUE: i32; const VALUE_2222: i32; }
            pub(crate) struct Bar {}
            impl Foo for Bar { const VALUE: i32 = 1;  const VALUE_2222: i32 = 2; }
            pub(crate) struct Bar2 {}
            impl Foo for Bar2 { const VALUE: i32 = 1;  const VALUE_2222: i32 = 2; }
            impl Bar2 { fn foo2() {} }
            fn main() {
                Bar::VALUE;
                Bar2::VALUE_2222;
            }
        "#]],
    );
}

#[test]
fn replace_path_in_different_contexts() {
    // Note the $0 inside module a::b which marks the point where the rule is interpreted. We
    // replace foo with bar, but both need different path qualifiers in different contexts. In f4,
    // foo is unqualified because of a use statement, however the replacement needs to be fully
    // qualified.
    assert_ssr_transform(
        "c::foo() ==>> c::bar()",
        r#"
            mod a {
                pub(crate) mod b {$0
                    pub(crate) mod c {
                        pub(crate) fn foo() {}
                        pub(crate) fn bar() {}
                        fn f1() { foo() }
                    }
                    fn f2() { c::foo() }
                }
                fn f3() { b::c::foo() }
            }
            use a::b::c::foo;
            fn f4() { foo() }
            "#,
        expect![[r#"
            mod a {
                pub(crate) mod b {
                    pub(crate) mod c {
                        pub(crate) fn foo() {}
                        pub(crate) fn bar() {}
                        fn f1() { bar() }
                    }
                    fn f2() { c::bar() }
                }
                fn f3() { b::c::bar() }
            }
            use a::b::c::foo;
            fn f4() { a::b::c::bar() }
            "#]],
    );
}

#[test]
fn replace_associated_function_with_generics() {
    assert_ssr_transform(
        "c::Foo::<$a>::new() ==>> d::Bar::<$a>::default()",
        r#"
            mod c {
                pub(crate) struct Foo<T> {v: T}
                impl<T> Foo<T> { pub(crate) fn new() {} }
                fn f1() {
                    Foo::<i32>::new();
                }
            }
            mod d {
                pub(crate) struct Bar<T> {v: T}
                impl<T> Bar<T> { pub(crate) fn default() {} }
                fn f1() {
                    super::c::Foo::<i32>::new();
                }
            }
            "#,
        expect![[r#"
            mod c {
                pub(crate) struct Foo<T> {v: T}
                impl<T> Foo<T> { pub(crate) fn new() {} }
                fn f1() {
                    crate::d::Bar::<i32>::default();
                }
            }
            mod d {
                pub(crate) struct Bar<T> {v: T}
                impl<T> Bar<T> { pub(crate) fn default() {} }
                fn f1() {
                    Bar::<i32>::default();
                }
            }
            "#]],
    );
}

#[test]
fn replace_type() {
    assert_ssr_transform(
        "Result<(), $a> ==>> Option<$a>",
        "struct Result<T, E> {} struct Option<T> {} fn f1() -> Result<(), Vec<Error>> {foo()}",
        expect![[
            "struct Result<T, E> {} struct Option<T> {} fn f1() -> Option<Vec<Error>> {foo()}"
        ]],
    );
    assert_ssr_transform(
        "dyn Trait<$a> ==>> DynTrait<$a>",
        r#"
trait Trait<T> {}
struct DynTrait<T> {}
fn f1() -> dyn Trait<Vec<Error>> {foo()}
"#,
        expect![[r#"
trait Trait<T> {}
struct DynTrait<T> {}
fn f1() -> DynTrait<Vec<Error>> {foo()}
"#]],
    );
}

#[test]
fn replace_macro_invocations() {
    assert_ssr_transform(
        "try_!($a) ==>> $a?",
        "macro_rules! try_ {() => {}} fn f1() -> Result<(), E> {bar(try_!(foo()));}",
        expect![["macro_rules! try_ {() => {}} fn f1() -> Result<(), E> {bar(foo()?);}"]],
    );
    // FIXME: Figure out why this doesn't work anymore
    // assert_ssr_transform(
    //     "foo!($a($b)) ==>> foo($b, $a)",
    //     "macro_rules! foo {() => {}} fn f1() {foo!(abc(def() + 2));}",
    //     expect![["macro_rules! foo {() => {}} fn f1() {foo(def() + 2, abc);}"]],
    // );
}

#[test]
fn replace_binary_op() {
    assert_ssr_transform(
        "$a + $b ==>> $b + $a",
        "fn f() {2 * 3 + 4 * 5}",
        expect![["fn f() {4 * 5 + 2 * 3}"]],
    );
    assert_ssr_transform(
        "$a + $b ==>> $b + $a",
        "fn f() {1 + 2 + 3 + 4}",
        expect![[r#"fn f() {4 + (3 + (2 + 1))}"#]],
    );
}

#[test]
fn match_binary_op() {
    assert_matches("$a + $b", "fn f() {1 + 2 + 3 + 4}", &["1 + 2", "1 + 2 + 3", "1 + 2 + 3 + 4"]);
}

#[test]
fn multiple_rules() {
    assert_ssr_transforms(
        &["$a + 1 ==>> add_one($a)", "$a + $b ==>> add($a, $b)"],
        "fn add() {} fn add_one() {} fn f() -> i32 {3 + 2 + 1}",
        expect![["fn add() {} fn add_one() {} fn f() -> i32 {add_one(add(3, 2))}"]],
    )
}

#[test]
fn multiple_rules_with_nested_matches() {
    assert_ssr_transforms(
        &["foo1($a) ==>> bar1($a)", "foo2($a) ==>> bar2($a)"],
        r#"
            fn foo1() {} fn foo2() {} fn bar1() {} fn bar2() {}
            fn f() {foo1(foo2(foo1(foo2(foo1(42)))))}
            "#,
        expect![[r#"
            fn foo1() {} fn foo2() {} fn bar1() {} fn bar2() {}
            fn f() {bar1(bar2(bar1(bar2(bar1(42)))))}
        "#]],
    )
}

#[test]
fn match_within_macro_invocation() {
    let code = r#"
            macro_rules! foo {
                ($a:stmt; $b:expr) => {
                    $b
                };
            }
            struct A {}
            impl A {
                fn bar() {}
            }
            fn f1() {
                let aaa = A {};
                foo!(macro_ignores_this(); aaa.bar());
            }
        "#;
    assert_matches("$a.bar()", code, &["aaa.bar()"]);
}

#[test]
fn replace_within_macro_expansion() {
    assert_ssr_transform(
        "$a.foo() ==>> bar($a)",
        r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn bar() {}
            fn f() {macro1!(5.x().foo().o2())}
            "#,
        expect![[r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn bar() {}
            fn f() {macro1!(bar(5.x()).o2())}
            "#]],
    )
}

#[test]
fn replace_outside_and_within_macro_expansion() {
    assert_ssr_transform(
        "foo($a) ==>> bar($a)",
        r#"
            fn foo() {} fn bar() {}
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {foo(foo(macro1!(foo(foo(42)))))}
            "#,
        expect![[r#"
            fn foo() {} fn bar() {}
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {bar(bar(macro1!(bar(bar(42)))))}
        "#]],
    )
}

#[test]
fn preserves_whitespace_within_macro_expansion() {
    assert_ssr_transform(
        "$a + $b ==>> $b - $a",
        r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {macro1!(1   *   2 + 3 + 4)}
            "#,
        expect![[r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {macro1!(4 - (3 - 1   *   2))}
            "#]],
    )
}

#[test]
fn add_parenthesis_when_necessary() {
    assert_ssr_transform(
        "foo($a) ==>> $a.to_string()",
        r#"
        fn foo(_: i32) {}
        fn bar3(v: i32) {
            foo(1 + 2);
            foo(-v);
        }
        "#,
        expect![[r#"
            fn foo(_: i32) {}
            fn bar3(v: i32) {
                (1 + 2).to_string();
                (-v).to_string();
            }
        "#]],
    )
}

#[test]
fn match_failure_reasons() {
    let code = r#"
        fn bar() {}
        macro_rules! foo {
            ($a:expr) => {
                1 + $a + 2
            };
        }
        fn f1() {
            bar(1, 2);
            foo!(5 + 43.to_string() + 5);
        }
        "#;
    assert_match_failure_reason(
        "bar($a, 3)",
        code,
        "bar(1, 2)",
        r#"Pattern wanted token '3' (INT_NUMBER), but code had token '2' (INT_NUMBER)"#,
    );
    assert_match_failure_reason(
        "42.to_string()",
        code,
        "43.to_string()",
        r#"Pattern wanted token '42' (INT_NUMBER), but code had token '43' (INT_NUMBER)"#,
    );
}

#[test]
fn overlapping_possible_matches() {
    // There are three possible matches here, however the middle one, `foo(foo(foo(42)))` shouldn't
    // match because it overlaps with the outer match. The inner match is permitted since it's is
    // contained entirely within the placeholder of the outer match.
    assert_matches(
        "foo(foo($a))",
        "fn foo() {} fn main() {foo(foo(foo(foo(42))))}",
        &["foo(foo(42))", "foo(foo(foo(foo(42))))"],
    );
}

#[test]
fn use_declaration_with_braces() {
    // It would be OK for a path rule to match and alter a use declaration. We shouldn't mess it up
    // though. In particular, we must not change `use foo::{baz, bar}` to `use foo::{baz,
    // foo2::bar2}`.
    cov_mark::check!(use_declaration_with_braces);
    assert_ssr_transform(
        "foo::bar ==>> foo2::bar2",
        r#"
        mod foo { pub(crate) fn bar() {} pub(crate) fn baz() {} }
        mod foo2 { pub(crate) fn bar2() {} }
        use foo::{baz, bar};
        fn main() { bar() }
        "#,
        expect![["
        mod foo { pub(crate) fn bar() {} pub(crate) fn baz() {} }
        mod foo2 { pub(crate) fn bar2() {} }
        use foo::{baz, bar};
        fn main() { foo2::bar2() }
        "]],
    )
}

#[test]
fn ufcs_matches_method_call() {
    let code = r#"
    struct Foo {}
    impl Foo {
        fn new(_: i32) -> Foo { Foo {} }
        fn do_stuff(&self, _: i32) {}
    }
    struct Bar {}
    impl Bar {
        fn new(_: i32) -> Bar { Bar {} }
        fn do_stuff(&self, v: i32) {}
    }
    fn main() {
        let b = Bar {};
        let f = Foo {};
        b.do_stuff(1);
        f.do_stuff(2);
        Foo::new(4).do_stuff(3);
        // Too many / too few args - should never match
        f.do_stuff(2, 10);
        f.do_stuff();
    }
    "#;
    assert_matches("Foo::do_stuff($a, $b)", code, &["f.do_stuff(2)", "Foo::new(4).do_stuff(3)"]);
    // The arguments needs special handling in the case of a function call matching a method call
    // and the first argument is different.
    assert_matches("Foo::do_stuff($a, 2)", code, &["f.do_stuff(2)"]);
    assert_matches("Foo::do_stuff(Foo::new(4), $b)", code, &["Foo::new(4).do_stuff(3)"]);

    assert_ssr_transform(
        "Foo::do_stuff(Foo::new($a), $b) ==>> Bar::new($b).do_stuff($a)",
        code,
        expect![[r#"
            struct Foo {}
            impl Foo {
                fn new(_: i32) -> Foo { Foo {} }
                fn do_stuff(&self, _: i32) {}
            }
            struct Bar {}
            impl Bar {
                fn new(_: i32) -> Bar { Bar {} }
                fn do_stuff(&self, v: i32) {}
            }
            fn main() {
                let b = Bar {};
                let f = Foo {};
                b.do_stuff(1);
                f.do_stuff(2);
                Bar::new(3).do_stuff(4);
                // Too many / too few args - should never match
                f.do_stuff(2, 10);
                f.do_stuff();
            }
        "#]],
    );
}

#[test]
fn pattern_is_a_single_segment_path() {
    cov_mark::check!(pattern_is_a_single_segment_path);
    // The first function should not be altered because the `foo` in scope at the cursor position is
    // a different `foo`. This case is special because "foo" can be parsed as a pattern (IDENT_PAT ->
    // NAME -> IDENT), which contains no path. If we're not careful we'll end up matching the `foo`
    // in `let foo` from the first function. Whether we should match the `let foo` in the second
    // function is less clear. At the moment, we don't. Doing so sounds like a rename operation,
    // which isn't really what SSR is for, especially since the replacement `bar` must be able to be
    // resolved, which means if we rename `foo` we'll get a name collision.
    assert_ssr_transform(
        "foo ==>> bar",
        r#"
        fn f1() -> i32 {
            let foo = 1;
            let bar = 2;
            foo
        }
        fn f1() -> i32 {
            let foo = 1;
            let bar = 2;
            foo$0
        }
        "#,
        expect![[r#"
            fn f1() -> i32 {
                let foo = 1;
                let bar = 2;
                foo
            }
            fn f1() -> i32 {
                let foo = 1;
                let bar = 2;
                bar
            }
        "#]],
    );
}

#[test]
fn replace_local_variable_reference() {
    // The pattern references a local variable `foo` in the block containing the cursor. We should
    // only replace references to this variable `foo`, not other variables that just happen to have
    // the same name.
    cov_mark::check!(cursor_after_semicolon);
    assert_ssr_transform(
        "foo + $a ==>> $a - foo",
        r#"
            fn bar1() -> i32 {
                let mut res = 0;
                let foo = 5;
                res += foo + 1;
                let foo = 10;
                res += foo + 2;$0
                res += foo + 3;
                let foo = 15;
                res += foo + 4;
                res
            }
            "#,
        expect![[r#"
            fn bar1() -> i32 {
                let mut res = 0;
                let foo = 5;
                res += foo + 1;
                let foo = 10;
                res += 2 - foo;
                res += 3 - foo;
                let foo = 15;
                res += foo + 4;
                res
            }
        "#]],
    )
}

#[test]
fn replace_path_within_selection() {
    assert_ssr_transform(
        "foo ==>> bar",
        r#"
        fn main() {
            let foo = 41;
            let bar = 42;
            do_stuff(foo);
            do_stuff(foo);$0
            do_stuff(foo);
            do_stuff(foo);$0
            do_stuff(foo);
        }"#,
        expect![[r#"
            fn main() {
                let foo = 41;
                let bar = 42;
                do_stuff(foo);
                do_stuff(foo);
                do_stuff(bar);
                do_stuff(bar);
                do_stuff(foo);
            }"#]],
    );
}

#[test]
fn replace_nonpath_within_selection() {
    cov_mark::check!(replace_nonpath_within_selection);
    assert_ssr_transform(
        "$a + $b ==>> $b * $a",
        r#"
        fn main() {
            let v = 1 + 2;$0
            let v2 = 3 + 3;
            let v3 = 4 + 5;$0
            let v4 = 6 + 7;
        }"#,
        expect![[r#"
            fn main() {
                let v = 1 + 2;
                let v2 = 3 * 3;
                let v3 = 5 * 4;
                let v4 = 6 + 7;
            }"#]],
    );
}

#[test]
fn replace_self() {
    // `foo(self)` occurs twice in the code, however only the first occurrence is the `self` that's
    // in scope where the rule is invoked.
    assert_ssr_transform(
        "foo(self) ==>> bar(self)",
        r#"
        struct S1 {}
        fn foo(_: &S1) {}
        fn bar(_: &S1) {}
        impl S1 {
            fn f1(&self) {
                foo(self)$0
            }
            fn f2(&self) {
                foo(self)
            }
        }
        "#,
        expect![[r#"
            struct S1 {}
            fn foo(_: &S1) {}
            fn bar(_: &S1) {}
            impl S1 {
                fn f1(&self) {
                    bar(self)
                }
                fn f2(&self) {
                    foo(self)
                }
            }
        "#]],
    );
}

#[test]
fn match_trait_method_call() {
    // `Bar::foo` and `Bar2::foo` resolve to the same function. Make sure we only match if the type
    // matches what's in the pattern. Also checks that we handle autoderef.
    let code = r#"
        pub(crate) struct Bar {}
        pub(crate) struct Bar2 {}
        pub(crate) trait Foo {
            fn foo(&self, _: i32) {}
        }
        impl Foo for Bar {}
        impl Foo for Bar2 {}
        fn main() {
            let v1 = Bar {};
            let v2 = Bar2 {};
            let v1_ref = &v1;
            let v2_ref = &v2;
            v1.foo(1);
            v2.foo(2);
            Bar::foo(&v1, 3);
            Bar2::foo(&v2, 4);
            v1_ref.foo(5);
            v2_ref.foo(6);
        }
        "#;
    assert_matches("Bar::foo($a, $b)", code, &["v1.foo(1)", "Bar::foo(&v1, 3)", "v1_ref.foo(5)"]);
    assert_matches("Bar2::foo($a, $b)", code, &["v2.foo(2)", "Bar2::foo(&v2, 4)", "v2_ref.foo(6)"]);
}

#[test]
fn replace_autoref_autoderef_capture() {
    // Here we have several calls to `$a.foo()`. In the first case autoref is applied, in the
    // second, we already have a reference, so it isn't. When $a is used in a context where autoref
    // doesn't apply, we need to prefix it with `&`. Finally, we have some cases where autoderef
    // needs to be applied.
    cov_mark::check!(replace_autoref_autoderef_capture);
    let code = r#"
        struct Foo {}
        impl Foo {
            fn foo(&self) {}
            fn foo2(&self) {}
        }
        fn bar(_: &Foo) {}
        fn main() {
            let f = Foo {};
            let fr = &f;
            let fr2 = &fr;
            let fr3 = &fr2;
            f.foo();
            fr.foo();
            fr2.foo();
            fr3.foo();
        }
        "#;
    assert_ssr_transform(
        "Foo::foo($a) ==>> bar($a)",
        code,
        expect![[r#"
            struct Foo {}
            impl Foo {
                fn foo(&self) {}
                fn foo2(&self) {}
            }
            fn bar(_: &Foo) {}
            fn main() {
                let f = Foo {};
                let fr = &f;
                let fr2 = &fr;
                let fr3 = &fr2;
                bar(&f);
                bar(&*fr);
                bar(&**fr2);
                bar(&***fr3);
            }
        "#]],
    );
    // If the placeholder is used as the receiver of another method call, then we don't need to
    // explicitly autoderef or autoref.
    assert_ssr_transform(
        "Foo::foo($a) ==>> $a.foo2()",
        code,
        expect![[r#"
            struct Foo {}
            impl Foo {
                fn foo(&self) {}
                fn foo2(&self) {}
            }
            fn bar(_: &Foo) {}
            fn main() {
                let f = Foo {};
                let fr = &f;
                let fr2 = &fr;
                let fr3 = &fr2;
                f.foo2();
                fr.foo2();
                fr2.foo2();
                fr3.foo2();
            }
        "#]],
    );
}

#[test]
fn replace_autoref_mut() {
    let code = r#"
        struct Foo {}
        impl Foo {
            fn foo(&mut self) {}
        }
        fn bar(_: &mut Foo) {}
        fn main() {
            let mut f = Foo {};
            f.foo();
            let fr = &mut f;
            fr.foo();
        }
        "#;
    assert_ssr_transform(
        "Foo::foo($a) ==>> bar($a)",
        code,
        expect![[r#"
            struct Foo {}
            impl Foo {
                fn foo(&mut self) {}
            }
            fn bar(_: &mut Foo) {}
            fn main() {
                let mut f = Foo {};
                bar(&mut f);
                let fr = &mut f;
                bar(&mut *fr);
            }
        "#]],
    );
}
