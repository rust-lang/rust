use rustc_span::edition::{DEFAULT_EDITION, Edition};

use super::{
    ErrorCodes, HeadingOffset, IdMap, Ignore, LangString, LangStringToken, Markdown,
    MarkdownItemInfo, TagIterator, find_testable_code, plain_text_summary, short_markdown_summary,
};

#[test]
fn test_unique_id() {
    let input = [
        "foo",
        "examples",
        "examples",
        "method.into_iter",
        "examples",
        "method.into_iter",
        "foo",
        "main-content",
        "search",
        "methods",
        "examples",
        "method.into_iter",
        "assoc_type.Item",
        "assoc_type.Item",
    ];
    let expected = [
        "foo",
        "examples",
        "examples-1",
        "method.into_iter",
        "examples-2",
        "method.into_iter-1",
        "foo-1",
        "main-content-1",
        "search-1",
        "methods",
        "examples-3",
        "method.into_iter-2",
        "assoc_type.Item",
        "assoc_type.Item-1",
    ];

    let mut map = IdMap::new();
    let actual: Vec<String> = input.iter().map(|s| map.derive(s)).collect();
    assert_eq!(&actual[..], expected);
}

#[test]
fn test_lang_string_parse() {
    fn t(lg: LangString) {
        let s = &lg.original;
        assert_eq!(LangString::parse(s, ErrorCodes::Yes, None), lg)
    }

    t(Default::default());
    t(LangString { original: "rust".into(), ..Default::default() });
    t(LangString {
        original: "rusta".into(),
        rust: false,
        unknown: vec!["rusta".into()],
        ..Default::default()
    });
    // error
    t(LangString { original: "{rust}".into(), rust: false, ..Default::default() });
    t(LangString {
        original: "{.rust}".into(),
        rust: true,
        added_classes: vec!["rust".into()],
        ..Default::default()
    });
    t(LangString {
        original: "custom,{.rust}".into(),
        rust: false,
        added_classes: vec!["rust".into()],
        ..Default::default()
    });
    t(LangString {
        original: "sh".into(),
        rust: false,
        unknown: vec!["sh".into()],
        ..Default::default()
    });
    t(LangString { original: "ignore".into(), ignore: Ignore::All, ..Default::default() });
    t(LangString {
        original: "ignore-foo".into(),
        ignore: Ignore::Some(vec!["foo".to_string()]),
        ..Default::default()
    });
    t(LangString { original: "should_panic".into(), should_panic: true, ..Default::default() });
    t(LangString { original: "no_run".into(), no_run: true, ..Default::default() });
    t(LangString { original: "test_harness".into(), test_harness: true, ..Default::default() });
    t(LangString {
        original: "compile_fail".into(),
        no_run: true,
        compile_fail: true,
        ..Default::default()
    });
    t(LangString {
        original: "no_run,example".into(),
        no_run: true,
        unknown: vec!["example".into()],
        ..Default::default()
    });
    t(LangString {
        original: "sh,should_panic".into(),
        should_panic: true,
        rust: false,
        unknown: vec!["sh".into()],
        ..Default::default()
    });
    t(LangString {
        original: "example,rust".into(),
        unknown: vec!["example".into()],
        ..Default::default()
    });
    t(LangString {
        original: "test_harness,rusta".into(),
        test_harness: true,
        unknown: vec!["rusta".into()],
        ..Default::default()
    });
    t(LangString {
        original: "text, no_run".into(),
        no_run: true,
        rust: false,
        unknown: vec!["text".into()],
        ..Default::default()
    });
    t(LangString {
        original: "text,no_run".into(),
        no_run: true,
        rust: false,
        unknown: vec!["text".into()],
        ..Default::default()
    });
    t(LangString {
        original: "text,no_run, ".into(),
        no_run: true,
        rust: false,
        unknown: vec!["text".into()],
        ..Default::default()
    });
    t(LangString {
        original: "text,no_run,".into(),
        no_run: true,
        rust: false,
        unknown: vec!["text".into()],
        ..Default::default()
    });
    t(LangString {
        original: "edition2015".into(),
        edition: Some(Edition::Edition2015),
        ..Default::default()
    });
    t(LangString {
        original: "edition2018".into(),
        edition: Some(Edition::Edition2018),
        ..Default::default()
    });
    t(LangString {
        original: "{class=test}".into(),
        added_classes: vec!["test".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: "custom,{class=test}".into(),
        added_classes: vec!["test".into()],
        rust: false,
        ..Default::default()
    });
    t(LangString {
        original: "{.test}".into(),
        added_classes: vec!["test".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: "custom,{.test}".into(),
        added_classes: vec!["test".into()],
        rust: false,
        ..Default::default()
    });
    t(LangString {
        original: "rust,{class=test,.test2}".into(),
        added_classes: vec!["test".into(), "test2".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: "{class=test:with:colon .test1}".into(),
        added_classes: vec!["test:with:colon".into(), "test1".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: "custom,{class=test:with:colon .test1}".into(),
        added_classes: vec!["test:with:colon".into(), "test1".into()],
        rust: false,
        ..Default::default()
    });
    t(LangString {
        original: "{class=first,class=second}".into(),
        added_classes: vec!["first".into(), "second".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: "custom,{class=first,class=second}".into(),
        added_classes: vec!["first".into(), "second".into()],
        rust: false,
        ..Default::default()
    });
    t(LangString {
        original: "{class=first,.second},unknown".into(),
        added_classes: vec!["first".into(), "second".into()],
        rust: false,
        unknown: vec!["unknown".into()],
        ..Default::default()
    });
    t(LangString {
        original: "{class=first .second} unknown".into(),
        added_classes: vec!["first".into(), "second".into()],
        rust: false,
        unknown: vec!["unknown".into()],
        ..Default::default()
    });
    // error
    t(LangString {
        original: "{.first.second}".into(),
        rust: true,
        added_classes: vec!["first.second".into()],
        ..Default::default()
    });
    // error
    t(LangString { original: "{class=first=second}".into(), rust: false, ..Default::default() });
    // error
    t(LangString {
        original: "{class=first.second}".into(),
        rust: true,
        added_classes: vec!["first.second".into()],
        ..Default::default()
    });
    // error
    t(LangString {
        original: "{class=.first}".into(),
        added_classes: vec![".first".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: r#"{class="first"}"#.into(),
        added_classes: vec!["first".into()],
        rust: true,
        ..Default::default()
    });
    t(LangString {
        original: r#"custom,{class="first"}"#.into(),
        added_classes: vec!["first".into()],
        rust: false,
        ..Default::default()
    });
    // error
    t(LangString { original: r#"{class=f"irst"}"#.into(), rust: false, ..Default::default() });
}

#[test]
fn test_lang_string_tokenizer() {
    fn case(lang_string: &str, want: &[LangStringToken<'_>]) {
        let have = TagIterator::new(lang_string, None).collect::<Vec<_>>();
        assert_eq!(have, want, "Unexpected lang string split for `{}`", lang_string);
    }

    case("", &[]);
    case("foo", &[LangStringToken::LangToken("foo")]);
    case("foo,bar", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case(".foo,.bar", &[]);
    case(
        "{.foo,.bar}",
        &[LangStringToken::ClassAttribute("foo"), LangStringToken::ClassAttribute("bar")],
    );
    case(
        "  {.foo,.bar}  ",
        &[LangStringToken::ClassAttribute("foo"), LangStringToken::ClassAttribute("bar")],
    );
    case("foo bar", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case("foo\tbar", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case("foo\t, bar", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case(" foo , bar ", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case(",,foo,,bar,,", &[LangStringToken::LangToken("foo"), LangStringToken::LangToken("bar")]);
    case("foo=bar", &[]);
    case("a-b-c", &[LangStringToken::LangToken("a-b-c")]);
    case("a_b_c", &[LangStringToken::LangToken("a_b_c")]);
}

#[test]
fn test_header() {
    fn t(input: &str, expect: &str) {
        let mut map = IdMap::new();
        let mut output = String::new();
        Markdown {
            content: input,
            links: &[],
            ids: &mut map,
            error_codes: ErrorCodes::Yes,
            edition: DEFAULT_EDITION,
            playground: &None,
            heading_offset: HeadingOffset::H2,
        }
        .write_into(&mut output)
        .unwrap();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(
        "# Foo bar",
        "<h2 id=\"foo-bar\"><a class=\"doc-anchor\" href=\"#foo-bar\">§</a>Foo bar</h2>",
    );
    t(
        "## Foo-bar_baz qux",
        "<h3 id=\"foo-bar_baz-qux\">\
             <a class=\"doc-anchor\" href=\"#foo-bar_baz-qux\">§</a>\
             Foo-bar_baz qux\
         </h3>",
    );
    t(
        "### **Foo** *bar* baz!?!& -_qux_-%",
        "<h4 id=\"foo-bar-baz--qux-\">\
            <a class=\"doc-anchor\" href=\"#foo-bar-baz--qux-\">§</a>\
            <strong>Foo</strong> <em>bar</em> baz!?!&amp; -<em>qux</em>-%\
         </h4>",
    );
    t(
        "#### **Foo?** & \\*bar?!*  _`baz`_ ❤ #qux",
        "<h5 id=\"foo--bar--baz--qux\">\
             <a class=\"doc-anchor\" href=\"#foo--bar--baz--qux\">§</a>\
             <strong>Foo?</strong> &amp; *bar?!*  <em><code>baz</code></em> ❤ #qux\
         </h5>",
    );
    t(
        "# Foo [bar](https://hello.yo)",
        "<h2 id=\"foo-bar\">\
             <a class=\"doc-anchor\" href=\"#foo-bar\">§</a>\
             Foo <a href=\"https://hello.yo\">bar</a>\
         </h2>",
    );
}

#[test]
fn test_header_ids_multiple_blocks() {
    let mut map = IdMap::new();
    fn t(map: &mut IdMap, input: &str, expect: &str) {
        let mut output = String::new();
        Markdown {
            content: input,
            links: &[],
            ids: map,
            error_codes: ErrorCodes::Yes,
            edition: DEFAULT_EDITION,
            playground: &None,
            heading_offset: HeadingOffset::H2,
        }
        .write_into(&mut output)
        .unwrap();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(
        &mut map,
        "# Example",
        "<h2 id=\"example\"><a class=\"doc-anchor\" href=\"#example\">§</a>Example</h2>",
    );
    t(
        &mut map,
        "# Panics",
        "<h2 id=\"panics\"><a class=\"doc-anchor\" href=\"#panics\">§</a>Panics</h2>",
    );
    t(
        &mut map,
        "# Example",
        "<h2 id=\"example-1\"><a class=\"doc-anchor\" href=\"#example-1\">§</a>Example</h2>",
    );
    t(
        &mut map,
        "# Search",
        "<h2 id=\"search-1\"><a class=\"doc-anchor\" href=\"#search-1\">§</a>Search</h2>",
    );
    t(
        &mut map,
        "# Example",
        "<h2 id=\"example-2\"><a class=\"doc-anchor\" href=\"#example-2\">§</a>Example</h2>",
    );
    t(
        &mut map,
        "# Panics",
        "<h2 id=\"panics-1\"><a class=\"doc-anchor\" href=\"#panics-1\">§</a>Panics</h2>",
    );
}

#[test]
fn test_short_markdown_summary() {
    fn t(input: &str, expect: &str) {
        let output = short_markdown_summary(input, &[][..]);
        assert_eq!(output, expect, "original: {}", input);
    }

    t("", "");
    t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
    t("*italic*", "<em>italic</em>");
    t("**bold**", "<strong>bold</strong>");
    t("Multi-line\nsummary", "Multi-line summary");
    t("Hard-break  \nsummary", "Hard-break summary");
    t("hello [Rust] :)\n\n[Rust]: https://www.rust-lang.org", "hello Rust :)");
    t("hello [Rust](https://www.rust-lang.org \"Rust\") :)", "hello Rust :)");
    t("dud [link]", "dud [link]");
    t("code `let x = i32;` ...", "code <code>let x = i32;</code> …");
    t("type `Type<'static>` ...", "type <code>Type&lt;&#39;static&gt;</code> …");
    // Test to ensure escaping and length-limiting work well together.
    // The output should be limited based on the input length,
    // rather than the output, because escaped versions of characters
    // are usually longer than how the character is actually displayed.
    t(
        "& & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & &",
        "&amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; \
         &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; &amp; \
         &amp; &amp; &amp; &amp; &amp; …",
    );
    t("# top header", "top header");
    t("# top header\n\nfollowed by a paragraph", "top header");
    t("## header", "header");
    t("first paragraph\n\nsecond paragraph", "first paragraph");
    t("```\nfn main() {}\n```", "");
    t("<div>hello</div>", "");
    t(
        "a *very*, **very** long first paragraph. it has lots of `inline code: Vec<T>`. and it has a [link](https://www.rust-lang.org).\nthat was a soft line break!  \nthat was a hard one\n\nsecond paragraph.",
        "a <em>very</em>, <strong>very</strong> long first paragraph. it has lots of …",
    );
}

#[test]
fn test_plain_text_summary() {
    fn t(input: &str, expect: &str) {
        let output = plain_text_summary(input, &[]);
        assert_eq!(output, expect, "original: {}", input);
    }

    t("", "");
    t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
    t("**bold**", "bold");
    t("Multi-line\nsummary", "Multi-line summary");
    t("Hard-break  \nsummary", "Hard-break summary");
    t("hello [Rust] :)\n\n[Rust]: https://www.rust-lang.org", "hello Rust :)");
    t("hello [Rust](https://www.rust-lang.org \"Rust\") :)", "hello Rust :)");
    t("dud [link]", "dud [link]");
    t("code `let x = i32;` ...", "code `let x = i32;` …");
    t("type `Type<'static>` ...", "type `Type<'static>` …");
    t("# top header", "top header");
    t("# top header\n\nfollowed by some text", "top header");
    t("## header", "header");
    t("first paragraph\n\nsecond paragraph", "first paragraph");
    t("```\nfn main() {}\n```", "");
    t("<div>hello</div>", "");
    t(
        "a *very*, **very** long first paragraph. it has lots of `inline code: Vec<T>`. and it has a [link](https://www.rust-lang.org).\nthat was a soft line break!  \nthat was a hard one\n\nsecond paragraph.",
        "a very, very long first paragraph. it has lots of `inline code: Vec<T>`. and it has a link. that was a soft line break! that was a hard one",
    );
}

#[test]
fn test_markdown_html_escape() {
    fn t(input: &str, expect: &str) {
        let mut idmap = IdMap::new();
        let mut output = String::new();
        MarkdownItemInfo(input, &mut idmap).write_into(&mut output).unwrap();
        assert_eq!(output, expect, "original: {}", input);
    }

    t("`Struct<'a, T>`", "<code>Struct&lt;'a, T&gt;</code>");
    t("Struct<'a, T>", "Struct&lt;’a, T&gt;");
    t("Struct<br>", "Struct&lt;br&gt;");
}

#[test]
fn test_find_testable_code_line() {
    fn t(input: &str, expect: &[usize]) {
        let mut lines = Vec::<usize>::new();
        find_testable_code(input, &mut lines, ErrorCodes::No, None);
        assert_eq!(lines, expect);
    }

    t("", &[]);
    t("```rust\n```", &[1]);
    t(" ```rust\n```", &[1]);
    t("\n```rust\n```", &[2]);
    t("\n ```rust\n```", &[2]);
    t("```rust\n```\n```rust\n```", &[1, 3]);
    t("```rust\n```\n ```rust\n```", &[1, 3]);
}

#[test]
fn test_ascii_with_prepending_hashtag() {
    fn t(input: &str, expect: &str) {
        let mut map = IdMap::new();
        let mut output = String::new();
        Markdown {
            content: input,
            links: &[],
            ids: &mut map,
            error_codes: ErrorCodes::Yes,
            edition: DEFAULT_EDITION,
            playground: &None,
            heading_offset: HeadingOffset::H2,
        }
        .write_into(&mut output)
        .unwrap();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(
        r#"```ascii
#..#.####.#....#.....##..
#..#.#....#....#....#..#.
####.###..#....#....#..#.
#..#.#....#....#....#..#.
#..#.#....#....#....#..#.
#..#.####.####.####..##..
```"#,
        "<div class=\"example-wrap\"><pre class=\"language-ascii\"><code>\
#..#.####.#....#.....##..
#..#.#....#....#....#..#.
####.###..#....#....#..#.
#..#.#....#....#....#..#.
#..#.#....#....#....#..#.
#..#.####.####.####..##..</code></pre></div>",
    );
    t(
        r#"```markdown
# hello
```"#,
        "<div class=\"example-wrap\"><pre class=\"language-markdown\"><code>\
# hello</code></pre></div>",
    );
}
