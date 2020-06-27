use crate::matching::MatchFailureReason;
use crate::{matching, Match, MatchFinder, SsrMatches, SsrPattern, SsrRule};
use matching::record_match_fails_reasons_scope;
use ra_db::{FileId, FileRange, SourceDatabaseExt};
use ra_syntax::ast::AstNode;
use ra_syntax::{ast, SyntaxKind, SyntaxNode, TextRange};

struct MatchDebugInfo {
    node: SyntaxNode,
    /// Our search pattern parsed as the same kind of syntax node as `node`. e.g. expression, item,
    /// etc. Will be absent if the pattern can't be parsed as that kind.
    pattern: Result<SyntaxNode, MatchFailureReason>,
    matched: Result<Match, MatchFailureReason>,
}

impl SsrPattern {
    pub(crate) fn tree_for_kind_with_reason(
        &self,
        kind: SyntaxKind,
    ) -> Result<&SyntaxNode, MatchFailureReason> {
        record_match_fails_reasons_scope(true, || self.tree_for_kind(kind))
            .map_err(|e| MatchFailureReason { reason: e.reason.unwrap() })
    }
}

impl std::fmt::Debug for MatchDebugInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "========= PATTERN ==========\n")?;
        match &self.pattern {
            Ok(pattern) => {
                write!(f, "{:#?}", pattern)?;
            }
            Err(err) => {
                write!(f, "{}", err.reason)?;
            }
        }
        write!(
            f,
            "\n============ AST ===========\n\
            {:#?}\n============================",
            self.node
        )?;
        match &self.matched {
            Ok(_) => write!(f, "Node matched")?,
            Err(reason) => write!(f, "Node failed to match because: {}", reason.reason)?,
        }
        Ok(())
    }
}

impl SsrMatches {
    /// Returns `self` with any nested matches removed and made into top-level matches.
    pub(crate) fn flattened(self) -> SsrMatches {
        let mut out = SsrMatches::default();
        self.flatten_into(&mut out);
        out
    }

    fn flatten_into(self, out: &mut SsrMatches) {
        for mut m in self.matches {
            for p in m.placeholder_values.values_mut() {
                std::mem::replace(&mut p.inner_matches, SsrMatches::default()).flatten_into(out);
            }
            out.matches.push(m);
        }
    }
}

impl Match {
    pub(crate) fn matched_text(&self) -> String {
        self.matched_node.text().to_string()
    }
}

impl<'db> MatchFinder<'db> {
    /// Adds a search pattern. For use if you intend to only call `find_matches_in_file`. If you
    /// intend to do replacement, use `add_rule` instead.
    fn add_search_pattern(&mut self, pattern: SsrPattern) {
        self.add_rule(SsrRule { pattern, template: "()".parse().unwrap() })
    }

    /// Finds all nodes in `file_id` whose text is exactly equal to `snippet` and attempts to match
    /// them, while recording reasons why they don't match. This API is useful for command
    /// line-based debugging where providing a range is difficult.
    fn debug_where_text_equal(&self, file_id: FileId, snippet: &str) -> Vec<MatchDebugInfo> {
        let file = self.sema.parse(file_id);
        let mut res = Vec::new();
        let file_text = self.sema.db.file_text(file_id);
        let mut remaining_text = file_text.as_str();
        let mut base = 0;
        let len = snippet.len() as u32;
        while let Some(offset) = remaining_text.find(snippet) {
            let start = base + offset as u32;
            let end = start + len;
            self.output_debug_for_nodes_at_range(
                file.syntax(),
                TextRange::new(start.into(), end.into()),
                &None,
                &mut res,
            );
            remaining_text = &remaining_text[offset + snippet.len()..];
            base = end;
        }
        res
    }

    fn output_debug_for_nodes_at_range(
        &self,
        node: &SyntaxNode,
        range: TextRange,
        restrict_range: &Option<FileRange>,
        out: &mut Vec<MatchDebugInfo>,
    ) {
        for node in node.children() {
            if !node.text_range().contains_range(range) {
                continue;
            }
            if node.text_range() == range {
                for rule in &self.rules {
                    let pattern =
                        rule.pattern.tree_for_kind_with_reason(node.kind()).map(|p| p.clone());
                    out.push(MatchDebugInfo {
                        matched: matching::get_match(true, rule, &node, restrict_range, &self.sema)
                            .map_err(|e| MatchFailureReason {
                                reason: e.reason.unwrap_or_else(|| {
                                    "Match failed, but no reason was given".to_owned()
                                }),
                            }),
                        pattern,
                        node: node.clone(),
                    });
                }
            } else if let Some(macro_call) = ast::MacroCall::cast(node.clone()) {
                if let Some(expanded) = self.sema.expand(&macro_call) {
                    if let Some(tt) = macro_call.token_tree() {
                        self.output_debug_for_nodes_at_range(
                            &expanded,
                            range,
                            &Some(self.sema.original_range(tt.syntax())),
                            out,
                        );
                    }
                }
            }
        }
    }
}

fn parse_error_text(query: &str) -> String {
    format!("{}", query.parse::<SsrRule>().unwrap_err())
}

#[test]
fn parser_empty_query() {
    assert_eq!(parse_error_text(""), "Parse error: Cannot find delemiter `==>>`");
}

#[test]
fn parser_no_delimiter() {
    assert_eq!(parse_error_text("foo()"), "Parse error: Cannot find delemiter `==>>`");
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
        // also get appplied to our expected result.
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
