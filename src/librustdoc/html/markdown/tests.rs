use super::{ErrorCodes, LangString, Markdown, MarkdownHtml, IdMap};
use super::plain_summary_line;
use std::cell::RefCell;
use syntax::edition::{Edition, DEFAULT_EDITION};

#[test]
fn test_lang_string_parse() {
    fn t(s: &str,
        should_panic: bool, no_run: bool, ignore: bool, rust: bool, test_harness: bool,
        compile_fail: bool, allow_fail: bool, error_codes: Vec<String>,
         edition: Option<Edition>) {
        assert_eq!(LangString::parse(s, ErrorCodes::Yes), LangString {
            should_panic,
            no_run,
            ignore,
            rust,
            test_harness,
            compile_fail,
            error_codes,
            original: s.to_owned(),
            allow_fail,
            edition,
        })
    }

    fn v() -> Vec<String> {
        Vec::new()
    }

    // ignore-tidy-linelength
    // marker                | should_panic | no_run | ignore | rust | test_harness
    //                       | compile_fail | allow_fail | error_codes | edition
    t("",                      false,         false,   false,   true,  false, false, false, v(), None);
    t("rust",                  false,         false,   false,   true,  false, false, false, v(), None);
    t("sh",                    false,         false,   false,   false, false, false, false, v(), None);
    t("ignore",                false,         false,   true,    true,  false, false, false, v(), None);
    t("should_panic",          true,          false,   false,   true,  false, false, false, v(), None);
    t("no_run",                false,         true,    false,   true,  false, false, false, v(), None);
    t("test_harness",          false,         false,   false,   true,  true,  false, false, v(), None);
    t("compile_fail",          false,         true,    false,   true,  false, true,  false, v(), None);
    t("allow_fail",            false,         false,   false,   true,  false, false, true,  v(), None);
    t("{.no_run .example}",    false,         true,    false,   true,  false, false, false, v(), None);
    t("{.sh .should_panic}",   true,          false,   false,   false, false, false, false, v(), None);
    t("{.example .rust}",      false,         false,   false,   true,  false, false, false, v(), None);
    t("{.test_harness .rust}", false,         false,   false,   true,  true,  false, false, v(), None);
    t("text, no_run",          false,         true,    false,   false, false, false, false, v(), None);
    t("text,no_run",           false,         true,    false,   false, false, false, false, v(), None);
    t("edition2015",           false,         false,   false,   true,  false, false, false, v(), Some(Edition::Edition2015));
    t("edition2018",           false,         false,   false,   true,  false, false, false, v(), Some(Edition::Edition2018));
}

#[test]
fn test_header() {
    fn t(input: &str, expect: &str) {
        let mut map = IdMap::new();
        let output = Markdown(input, &[], RefCell::new(&mut map),
                              ErrorCodes::Yes, DEFAULT_EDITION).to_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t("# Foo bar", "<h1 id=\"foo-bar\" class=\"section-header\">\
      <a href=\"#foo-bar\">Foo bar</a></h1>");
    t("## Foo-bar_baz qux", "<h2 id=\"foo-bar_baz-qux\" class=\"section-\
      header\"><a href=\"#foo-bar_baz-qux\">Foo-bar_baz qux</a></h2>");
    t("### **Foo** *bar* baz!?!& -_qux_-%",
      "<h3 id=\"foo-bar-baz--qux-\" class=\"section-header\">\
      <a href=\"#foo-bar-baz--qux-\"><strong>Foo</strong> \
      <em>bar</em> baz!?!&amp; -<em>qux</em>-%</a></h3>");
    t("#### **Foo?** & \\*bar?!*  _`baz`_ ❤ #qux",
      "<h4 id=\"foo--bar--baz--qux\" class=\"section-header\">\
      <a href=\"#foo--bar--baz--qux\"><strong>Foo?</strong> &amp; *bar?!*  \
      <em><code>baz</code></em> ❤ #qux</a></h4>");
}

#[test]
fn test_header_ids_multiple_blocks() {
    let mut map = IdMap::new();
    fn t(map: &mut IdMap, input: &str, expect: &str) {
        let output = Markdown(input, &[], RefCell::new(map),
                              ErrorCodes::Yes, DEFAULT_EDITION).to_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t(&mut map, "# Example", "<h1 id=\"example\" class=\"section-header\">\
        <a href=\"#example\">Example</a></h1>");
    t(&mut map, "# Panics", "<h1 id=\"panics\" class=\"section-header\">\
        <a href=\"#panics\">Panics</a></h1>");
    t(&mut map, "# Example", "<h1 id=\"example-1\" class=\"section-header\">\
        <a href=\"#example-1\">Example</a></h1>");
    t(&mut map, "# Main", "<h1 id=\"main\" class=\"section-header\">\
        <a href=\"#main\">Main</a></h1>");
    t(&mut map, "# Example", "<h1 id=\"example-2\" class=\"section-header\">\
        <a href=\"#example-2\">Example</a></h1>");
    t(&mut map, "# Panics", "<h1 id=\"panics-1\" class=\"section-header\">\
        <a href=\"#panics-1\">Panics</a></h1>");
}

#[test]
fn test_plain_summary_line() {
    fn t(input: &str, expect: &str) {
        let output = plain_summary_line(input);
        assert_eq!(output, expect, "original: {}", input);
    }

    t("hello [Rust](https://www.rust-lang.org) :)", "hello Rust :)");
    t("hello [Rust](https://www.rust-lang.org \"Rust\") :)", "hello Rust :)");
    t("code `let x = i32;` ...", "code `let x = i32;` ...");
    t("type `Type<'static>` ...", "type `Type<'static>` ...");
    t("# top header", "top header");
    t("## header", "header");
}

#[test]
fn test_markdown_html_escape() {
    fn t(input: &str, expect: &str) {
        let mut idmap = IdMap::new();
        let output = MarkdownHtml(input, RefCell::new(&mut idmap),
                                  ErrorCodes::Yes, DEFAULT_EDITION).to_string();
        assert_eq!(output, expect, "original: {}", input);
    }

    t("`Struct<'a, T>`", "<p><code>Struct&lt;'a, T&gt;</code></p>\n");
    t("Struct<'a, T>", "<p>Struct&lt;'a, T&gt;</p>\n");
    t("Struct<br>", "<p>Struct&lt;br&gt;</p>\n");
}
