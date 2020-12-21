use super::{plain_text_summary, short_markdown_summary};
use super::{ErrorCodes, IdMap, Ignore, LangString, Markdown, MarkdownHtml};
use rustc_span::edition::{Edition, DEFAULT_EDITION};
use std::cell::RefCell;

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
        "main",
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
        "main-1",
        "search-1",
        "methods",
        "examples-3",
        "method.into_iter-2",
        "assoc_type.Item",
        "assoc_type.Item-1",
    ];

    let map = RefCell::new(IdMap::new());
    let test = || {
        let mut map = map.borrow_mut();
        let actual: Vec<String> = input.iter().map(|s| map.derive(s.to_string())).collect();
        assert_eq!(&actual[..], expected);
    };
    test();
    map.borrow_mut().reset();
    test();
}

#[test]
fn test_lang_string_parse() {
    fn t(lg: LangString) {
        let s = &lg.original;
        assert_eq!(LangString::parse(s, ErrorCodes::Yes, true, None), lg)
    }

    t(Default::default());
    t(LangString { original: "rust".into(), ..Default::default() });
    t(LangString { original: "sh".into(), rust: false, ..Default::default() });
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
    t(LangString { original: "allow_fail".into(), allow_fail: true, ..Default::default() });
    t(LangString { original: "{.no_run .example}".into(), no_run: true, ..Default::default() });
    t(LangString {
        original: "{.sh .should_panic}".into(),
        should_panic: true,
        rust: false,
        ..Default::default()
    });
    t(LangString { original: "{.example .rust}".into(), ..Default::default() });
    t(LangString {
        original: "{.test_harness .rust}".into(),
        test_harness: true,
        ..Default::default()
    });
    t(LangString {
        original: "text, no_run".into(),
        no_run: true,
        rust: false,
        ..Default::default()
    });
    t(LangString {
        original: "text,no_run".into(),
        no_run: true,
        rust: false,
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
}

#[test]
fn test_header() {
    fn t(input: &str, expect: &str) {
        let mut map = IdMap::new();
        let output =
            Markdown(input, &[], &mut map, ErrorCodes::Yes, DEFAULT_EDITION, &None).into_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(
        "# Foo bar",
        "<h1 id=\"foo-bar\" class=\"section-header\"><a href=\"#foo-bar\">Foo bar</a></h1>",
    );
    t(
        "## Foo-bar_baz qux",
        "<h2 id=\"foo-bar_baz-qux\" class=\"section-header\">\
         <a href=\"#foo-bar_baz-qux\">Foo-bar_baz qux</a></h2>",
    );
    t(
        "### **Foo** *bar* baz!?!& -_qux_-%",
        "<h3 id=\"foo-bar-baz--qux-\" class=\"section-header\">\
            <a href=\"#foo-bar-baz--qux-\"><strong>Foo</strong> \
            <em>bar</em> baz!?!&amp; -<em>qux</em>-%</a>\
         </h3>",
    );
    t(
        "#### **Foo?** & \\*bar?!*  _`baz`_ ❤ #qux",
        "<h4 id=\"foo--bar--baz--qux\" class=\"section-header\">\
             <a href=\"#foo--bar--baz--qux\"><strong>Foo?</strong> &amp; *bar?!*  \
             <em><code>baz</code></em> ❤ #qux</a>\
         </h4>",
    );
}

#[test]
fn test_header_ids_multiple_blocks() {
    let mut map = IdMap::new();
    fn t(map: &mut IdMap, input: &str, expect: &str) {
        let output =
            Markdown(input, &[], map, ErrorCodes::Yes, DEFAULT_EDITION, &None).into_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(
        &mut map,
        "# Example",
        "<h1 id=\"example\" class=\"section-header\"><a href=\"#example\">Example</a></h1>",
    );
    t(
        &mut map,
        "# Panics",
        "<h1 id=\"panics\" class=\"section-header\"><a href=\"#panics\">Panics</a></h1>",
    );
    t(
        &mut map,
        "# Example",
        "<h1 id=\"example-1\" class=\"section-header\"><a href=\"#example-1\">Example</a></h1>",
    );
    t(
        &mut map,
        "# Main",
        "<h1 id=\"main-1\" class=\"section-header\"><a href=\"#main-1\">Main</a></h1>",
    );
    t(
        &mut map,
        "# Example",
        "<h1 id=\"example-2\" class=\"section-header\"><a href=\"#example-2\">Example</a></h1>",
    );
    t(
        &mut map,
        "# Panics",
        "<h1 id=\"panics-1\" class=\"section-header\"><a href=\"#panics-1\">Panics</a></h1>",
    );
}

#[test]
fn test_short_markdown_summary() {
    fn t(input: &str, expect: &str) {
        let output = short_markdown_summary(input);
        assert_eq!(output, expect, "original: {}", input);
    }

    t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
    t("*italic*", "<em>italic</em>");
    t("**bold**", "<strong>bold</strong>");
    t("Multi-line\nsummary", "Multi-line summary");
    t("Hard-break  \nsummary", "Hard-break summary");
    t("hello [Rust] :)\n\n[Rust]: https://www.rust-lang.org", "hello Rust :)");
    t("hello [Rust](https://www.rust-lang.org \"Rust\") :)", "hello Rust :)");
    t("code `let x = i32;` ...", "code <code>let x = i32;</code> ...");
    t("type `Type<'static>` ...", "type <code>Type<'static></code> ...");
    t("# top header", "top header");
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
        let output = plain_text_summary(input);
        assert_eq!(output, expect, "original: {}", input);
    }

    t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
    t("**bold**", "bold");
    t("Multi-line\nsummary", "Multi-line summary");
    t("Hard-break  \nsummary", "Hard-break summary");
    t("hello [Rust] :)\n\n[Rust]: https://www.rust-lang.org", "hello Rust :)");
    t("hello [Rust](https://www.rust-lang.org \"Rust\") :)", "hello Rust :)");
    t("code `let x = i32;` ...", "code `let x = i32;` ...");
    t("type `Type<'static>` ...", "type `Type<'static>` ...");
    t("# top header", "top header");
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
        let output =
            MarkdownHtml(input, &mut idmap, ErrorCodes::Yes, DEFAULT_EDITION, &None).into_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t("`Struct<'a, T>`", "<p><code>Struct&lt;'a, T&gt;</code></p>\n");
    t("Struct<'a, T>", "<p>Struct&lt;'a, T&gt;</p>\n");
    t("Struct<br>", "<p>Struct&lt;br&gt;</p>\n");
}
