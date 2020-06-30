use crate::{MatchFinder, SsrRule};
use ra_db::{FileId, SourceDatabaseExt};

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
        "Parse error: Name `a` repeats more than once"
    );
}

#[test]
fn parser_invalid_pattern() {
    assert_eq!(
        parse_error_text(" ==>> ()"),
        "Parse error: Pattern is not a valid Rust expression, type, item, path or pattern"
    );
}

#[test]
fn parser_invalid_template() {
    assert_eq!(
        parse_error_text("() ==>> )"),
        "Parse error: Replacement is not a valid Rust expression, type, item, path or pattern"
    );
}

#[test]
fn parser_undefined_placeholder_in_replacement() {
    assert_eq!(
        parse_error_text("42 ==>> $a"),
        "Parse error: Replacement contains undefined placeholders: $a"
    );
}

fn single_file(code: &str) -> (ra_ide_db::RootDatabase, FileId) {
    use ra_db::fixture::WithFixture;
    ra_ide_db::RootDatabase::with_single_file(code)
}

fn assert_ssr_transform(rule: &str, input: &str, result: &str) {
    assert_ssr_transforms(&[rule], input, result);
}

fn normalize_code(code: &str) -> String {
    let (db, file_id) = single_file(code);
    db.file_text(file_id).to_string()
}

fn assert_ssr_transforms(rules: &[&str], input: &str, result: &str) {
    let (db, file_id) = single_file(input);
    let mut match_finder = MatchFinder::new(&db);
    for rule in rules {
        let rule: SsrRule = rule.parse().unwrap();
        match_finder.add_rule(rule);
    }
    if let Some(edits) = match_finder.edits_for_file(file_id) {
        // Note, db.file_text is not necessarily the same as `input`, since fixture parsing alters
        // stuff.
        let mut after = db.file_text(file_id).to_string();
        edits.apply(&mut after);
        // Likewise, we need to make sure that whatever transformations fixture parsing applies,
        // also get applied to our expected result.
        let result = normalize_code(result);
        assert_eq!(after, result);
    } else {
        panic!("No edits were made");
    }
}

fn assert_matches(pattern: &str, code: &str, expected: &[&str]) {
    let (db, file_id) = single_file(code);
    let mut match_finder = MatchFinder::new(&db);
    match_finder.add_search_pattern(pattern.parse().unwrap());
    let matched_strings: Vec<String> = match_finder
        .find_matches_in_file(file_id)
        .flattened()
        .matches
        .iter()
        .map(|m| m.matched_text())
        .collect();
    if matched_strings != expected && !expected.is_empty() {
        let debug_info = match_finder.debug_where_text_equal(file_id, &expected[0]);
        eprintln!("Test is about to fail. Some possibly useful info: {} nodes had text exactly equal to '{}'", debug_info.len(), &expected[0]);
        for d in debug_info {
            eprintln!("{:#?}", d);
        }
    }
    assert_eq!(matched_strings, expected);
}

fn assert_no_match(pattern: &str, code: &str) {
    assert_matches(pattern, code, &[]);
}

fn assert_match_failure_reason(pattern: &str, code: &str, snippet: &str, expected_reason: &str) {
    let (db, file_id) = single_file(code);
    let mut match_finder = MatchFinder::new(&db);
    match_finder.add_search_pattern(pattern.parse().unwrap());
    let mut reasons = Vec::new();
    for d in match_finder.debug_where_text_equal(file_id, snippet) {
        if let Some(reason) = d.match_failure_reason() {
            reasons.push(reason.to_owned());
        }
    }
    assert_eq!(reasons, vec![expected_reason]);
}

#[test]
fn ssr_function_to_method() {
    assert_ssr_transform(
        "my_function($a, $b) ==>> ($a).my_method($b)",
        "loop { my_function( other_func(x, y), z + w) }",
        "loop { (other_func(x, y)).my_method(z + w) }",
    )
}

#[test]
fn ssr_nested_function() {
    assert_ssr_transform(
        "foo($a, $b, $c) ==>> bar($c, baz($a, $b))",
        "fn main { foo  (x + value.method(b), x+y-z, true && false) }",
        "fn main { bar(true && false, baz(x + value.method(b), x+y-z)) }",
    )
}

#[test]
fn ssr_expected_spacing() {
    assert_ssr_transform(
        "foo($x) + bar() ==>> bar($x)",
        "fn main() { foo(5) + bar() }",
        "fn main() { bar(5) }",
    );
}

#[test]
fn ssr_with_extra_space() {
    assert_ssr_transform(
        "foo($x  ) +    bar() ==>> bar($x)",
        "fn main() { foo(  5 )  +bar(   ) }",
        "fn main() { bar(5) }",
    );
}

#[test]
fn ssr_keeps_nested_comment() {
    assert_ssr_transform(
        "foo($x) ==>> bar($x)",
        "fn main() { foo(other(5 /* using 5 */)) }",
        "fn main() { bar(other(5 /* using 5 */)) }",
    )
}

#[test]
fn ssr_keeps_comment() {
    assert_ssr_transform(
        "foo($x) ==>> bar($x)",
        "fn main() { foo(5 /* using 5 */) }",
        "fn main() { bar(5)/* using 5 */ }",
    )
}

#[test]
fn ssr_struct_lit() {
    assert_ssr_transform(
        "foo{a: $a, b: $b} ==>> foo::new($a, $b)",
        "fn main() { foo{b:2, a:1} }",
        "fn main() { foo::new(1, 2) }",
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
    assert_matches(
        "struct $n {$f: Option<String>}",
        "struct Bar {} struct Foo {name: Option<String>}",
        &["struct Foo {name: Option<String>}"],
    );
}

#[test]
fn match_expr() {
    let code = "fn f() -> i32 {foo(40 + 2, 42)}";
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
    let code = "fn f() -> i32 {foo(bar(40, 2), 42)}";
    assert_matches("foo($a, $b)", code, &["foo(bar(40, 2), 42)"]);
    assert_no_match("foo($a, $b, $c)", code);
    assert_no_match("foo($a)", code);
    assert_matches("bar($a, $b)", code, &["bar(40, 2)"]);
}

// Trailing commas in the code should be ignored.
#[test]
fn match_with_trailing_commas() {
    // Code has comma, pattern doesn't.
    assert_matches("foo($a, $b)", "fn f() {foo(1, 2,);}", &["foo(1, 2,)"]);
    assert_matches("Foo{$a, $b}", "fn f() {Foo{1, 2,};}", &["Foo{1, 2,}"]);

    // Pattern has comma, code doesn't.
    assert_matches("foo($a, $b,)", "fn f() {foo(1, 2);}", &["foo(1, 2)"]);
    assert_matches("Foo{$a, $b,}", "fn f() {Foo{1, 2};}", &["Foo{1, 2}"]);
}

#[test]
fn match_type() {
    assert_matches("i32", "fn f() -> i32 {1  +  2}", &["i32"]);
    assert_matches("Option<$a>", "fn f() -> Option<i32> {42}", &["Option<i32>"]);
    assert_no_match("Option<$a>", "fn f() -> Result<i32, ()> {42}");
}

#[test]
fn match_struct_instantiation() {
    assert_matches(
        "Foo {bar: 1, baz: 2}",
        "fn f() {Foo {bar: 1, baz: 2}}",
        &["Foo {bar: 1, baz: 2}"],
    );
    // Now with placeholders for all parts of the struct.
    assert_matches(
        "Foo {$a: $b, $c: $d}",
        "fn f() {Foo {bar: 1, baz: 2}}",
        &["Foo {bar: 1, baz: 2}"],
    );
    assert_matches("Foo {}", "fn f() {Foo {}}", &["Foo {}"]);
}

#[test]
fn match_path() {
    assert_matches("foo::bar", "fn f() {foo::bar(42)}", &["foo::bar"]);
    assert_matches("$a::bar", "fn f() {foo::bar(42)}", &["foo::bar"]);
    assert_matches("foo::$b", "fn f() {foo::bar(42)}", &["foo::bar"]);
}

#[test]
fn match_pattern() {
    assert_matches("Some($a)", "fn f() {if let Some(x) = foo() {}}", &["Some(x)"]);
}

#[test]
fn match_reordered_struct_instantiation() {
    assert_matches(
        "Foo {aa: 1, b: 2, ccc: 3}",
        "fn f() {Foo {b: 2, ccc: 3, aa: 1}}",
        &["Foo {b: 2, ccc: 3, aa: 1}"],
    );
    assert_no_match("Foo {a: 1}", "fn f() {Foo {b: 1}}");
    assert_no_match("Foo {a: 1}", "fn f() {Foo {a: 2}}");
    assert_no_match("Foo {a: 1, b: 2}", "fn f() {Foo {a: 1}}");
    assert_no_match("Foo {a: 1, b: 2}", "fn f() {Foo {b: 2}}");
    assert_no_match("Foo {a: 1, }", "fn f() {Foo {a: 1, b: 2}}");
    assert_no_match("Foo {a: 1, z: 9}", "fn f() {Foo {a: 1}}");
}

#[test]
fn match_macro_invocation() {
    assert_matches("foo!($a)", "fn() {foo(foo!(foo()))}", &["foo!(foo())"]);
    assert_matches("foo!(41, $a, 43)", "fn() {foo!(41, 42, 43)}", &["foo!(41, 42, 43)"]);
    assert_no_match("foo!(50, $a, 43)", "fn() {foo!(41, 42, 43}");
    assert_no_match("foo!(41, $a, 50)", "fn() {foo!(41, 42, 43}");
    assert_matches("foo!($a())", "fn() {foo!(bar())}", &["foo!(bar())"]);
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
    assert_ssr_transform("foo() ==>> bar()", "fn f1() {foo(); foo();}", "fn f1() {bar(); bar();}");
}

#[test]
fn replace_function_call_with_placeholders() {
    assert_ssr_transform(
        "foo($a, $b) ==>> bar($b, $a)",
        "fn f1() {foo(5, 42)}",
        "fn f1() {bar(42, 5)}",
    );
}

#[test]
fn replace_nested_function_calls() {
    assert_ssr_transform(
        "foo($a) ==>> bar($a)",
        "fn f1() {foo(foo(42))}",
        "fn f1() {bar(bar(42))}",
    );
}

#[test]
fn replace_type() {
    assert_ssr_transform(
        "Result<(), $a> ==>> Option<$a>",
        "fn f1() -> Result<(), Vec<Error>> {foo()}",
        "fn f1() -> Option<Vec<Error>> {foo()}",
    );
}

#[test]
fn replace_struct_init() {
    assert_ssr_transform(
        "Foo {a: $a, b: $b} ==>> Foo::new($a, $b)",
        "fn f1() {Foo{b: 1, a: 2}}",
        "fn f1() {Foo::new(2, 1)}",
    );
}

#[test]
fn replace_macro_invocations() {
    assert_ssr_transform(
        "try!($a) ==>> $a?",
        "fn f1() -> Result<(), E> {bar(try!(foo()));}",
        "fn f1() -> Result<(), E> {bar(foo()?);}",
    );
    assert_ssr_transform(
        "foo!($a($b)) ==>> foo($b, $a)",
        "fn f1() {foo!(abc(def() + 2));}",
        "fn f1() {foo(def() + 2, abc);}",
    );
}

#[test]
fn replace_binary_op() {
    assert_ssr_transform(
        "$a + $b ==>> $b + $a",
        "fn f() {2 * 3 + 4 * 5}",
        "fn f() {4 * 5 + 2 * 3}",
    );
    assert_ssr_transform(
        "$a + $b ==>> $b + $a",
        "fn f() {1 + 2 + 3 + 4}",
        "fn f() {4 + 3 + 2 + 1}",
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
        "fn f() -> i32 {3 + 2 + 1}",
        "fn f() -> i32 {add_one(add(3, 2))}",
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
            fn f() {macro1!(5.x().foo().o2())}"#,
        r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {macro1!(bar(5.x()).o2())}"#,
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
            fn f() {macro1!(1   *   2 + 3 + 4}"#,
        r#"
            macro_rules! macro1 {
                ($a:expr) => {$a}
            }
            fn f() {macro1!(4 - 3 - 1   *   2}"#,
    )
}

#[test]
fn match_failure_reasons() {
    let code = r#"
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
