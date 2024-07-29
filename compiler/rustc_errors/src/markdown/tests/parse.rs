use ParseOpt as PO;

use super::*;

#[test]
fn test_parse_simple() {
    let buf = "**abcd** rest";
    let (t, r) = parse_simple_pat(buf.as_bytes(), b"**", b"**", PO::None, MdTree::Strong).unwrap();
    assert_eq!(t, MdTree::Strong("abcd"));
    assert_eq!(r, b" rest");

    // Escaping should fail
    let buf = r"**abcd\** rest";
    let res = parse_simple_pat(buf.as_bytes(), b"**", b"**", PO::None, MdTree::Strong);
    assert!(res.is_none());
}

#[test]
fn test_parse_comment() {
    let opt = PO::TrimNoEsc;
    let buf = "<!-- foobar! -->rest";
    let (t, r) = parse_simple_pat(buf.as_bytes(), CMT_S, CMT_E, opt, MdTree::Comment).unwrap();
    assert_eq!(t, MdTree::Comment("foobar!"));
    assert_eq!(r, b"rest");

    let buf = r"<!-- foobar! \-->rest";
    let (t, r) = parse_simple_pat(buf.as_bytes(), CMT_S, CMT_E, opt, MdTree::Comment).unwrap();
    assert_eq!(t, MdTree::Comment(r"foobar! \"));
    assert_eq!(r, b"rest");
}

#[test]
fn test_parse_heading() {
    let buf1 = "# Top level\nrest";
    let (t, r) = parse_heading(buf1.as_bytes()).unwrap();
    assert_eq!(t, MdTree::Heading(1, vec![MdTree::PlainText("Top level")].into()));
    assert_eq!(r, b"\nrest");

    let buf1 = "# Empty";
    let (t, r) = parse_heading(buf1.as_bytes()).unwrap();
    assert_eq!(t, MdTree::Heading(1, vec![MdTree::PlainText("Empty")].into()));
    assert_eq!(r, b"");

    // Combo
    let buf2 = "### Top `level` _woo_\nrest";
    let (t, r) = parse_heading(buf2.as_bytes()).unwrap();
    assert_eq!(
        t,
        MdTree::Heading(
            3,
            vec![
                MdTree::PlainText("Top "),
                MdTree::CodeInline("level"),
                MdTree::PlainText(" "),
                MdTree::Emphasis("woo"),
            ]
            .into()
        )
    );
    assert_eq!(r, b"\nrest");
}

#[test]
fn test_parse_code_inline() {
    let buf1 = "`abcd` rest";
    let (t, r) = parse_codeinline(buf1.as_bytes()).unwrap();
    assert_eq!(t, MdTree::CodeInline("abcd"));
    assert_eq!(r, b" rest");

    // extra backticks, newline
    let buf2 = "```ab\ncd``` rest";
    let (t, r) = parse_codeinline(buf2.as_bytes()).unwrap();
    assert_eq!(t, MdTree::CodeInline("ab\ncd"));
    assert_eq!(r, b" rest");

    // test no escaping
    let buf3 = r"`abcd\` rest";
    let (t, r) = parse_codeinline(buf3.as_bytes()).unwrap();
    assert_eq!(t, MdTree::CodeInline(r"abcd\"));
    assert_eq!(r, b" rest");
}

#[test]
fn test_parse_code_block() {
    let buf1 = "```rust\ncode\ncode\n```\nleftovers";
    let (t, r) = parse_codeblock(buf1.as_bytes());
    assert_eq!(t, MdTree::CodeBlock { txt: "code\ncode", lang: Some("rust") });
    assert_eq!(r, b"\nleftovers");

    let buf2 = "`````\ncode\ncode````\n`````\nleftovers";
    let (t, r) = parse_codeblock(buf2.as_bytes());
    assert_eq!(t, MdTree::CodeBlock { txt: "code\ncode````", lang: None });
    assert_eq!(r, b"\nleftovers");
}

#[test]
fn test_parse_link() {
    let simple = "[see here](docs.rs) other";
    let (t, r) = parse_any_link(simple.as_bytes(), false).unwrap();
    assert_eq!(t, MdTree::Link { disp: "see here", link: "docs.rs" });
    assert_eq!(r, b" other");

    let simple_toplevel = "[see here](docs.rs) other";
    let (t, r) = parse_any_link(simple_toplevel.as_bytes(), true).unwrap();
    assert_eq!(t, MdTree::Link { disp: "see here", link: "docs.rs" });
    assert_eq!(r, b" other");

    let reference = "[see here] other";
    let (t, r) = parse_any_link(reference.as_bytes(), true).unwrap();
    assert_eq!(t, MdTree::RefLink { disp: "see here", id: None });
    assert_eq!(r, b" other");

    let reference_full = "[see here][docs-rs] other";
    let (t, r) = parse_any_link(reference_full.as_bytes(), false).unwrap();
    assert_eq!(t, MdTree::RefLink { disp: "see here", id: Some("docs-rs") });
    assert_eq!(r, b" other");

    let reference_def = "[see here]: docs.rs\nother";
    let (t, r) = parse_any_link(reference_def.as_bytes(), true).unwrap();
    assert_eq!(t, MdTree::LinkDef { id: "see here", link: "docs.rs" });
    assert_eq!(r, b"\nother");
}

const IND1: &str = r"test standard
    ind
    ind2
not ind";
const IND2: &str = r"test end of stream
  1
  2
";
const IND3: &str = r"test empty lines
  1
  2

not ind";

#[test]
fn test_indented_section() {
    let (t, r) = get_indented_section(IND1.as_bytes());
    assert_eq!(str::from_utf8(t).unwrap(), "test standard\n    ind\n    ind2");
    assert_eq!(str::from_utf8(r).unwrap(), "\nnot ind");

    let (txt, rest) = get_indented_section(IND2.as_bytes());
    assert_eq!(str::from_utf8(txt).unwrap(), "test end of stream\n  1\n  2\n");
    assert_eq!(str::from_utf8(rest).unwrap(), "");

    let (txt, rest) = get_indented_section(IND3.as_bytes());
    assert_eq!(str::from_utf8(txt).unwrap(), "test empty lines\n  1\n  2\n");
    assert_eq!(str::from_utf8(rest).unwrap(), "\nnot ind");
}

const HBT: &str = r"# Heading

content";

#[test]
fn test_heading_breaks() {
    let expected = vec![
        MdTree::Heading(1, vec![MdTree::PlainText("Heading")].into()),
        MdTree::PlainText("content"),
    ]
    .into();
    let res = entrypoint(HBT);
    assert_eq!(res, expected);
}

const NL1: &str = r"start

end";
const NL2: &str = r"start


end";
const NL3: &str = r"start



end";

#[test]
fn test_newline_breaks() {
    let expected =
        vec![MdTree::PlainText("start"), MdTree::ParagraphBreak, MdTree::PlainText("end")].into();
    for (idx, check) in [NL1, NL2, NL3].iter().enumerate() {
        let res = entrypoint(check);
        assert_eq!(res, expected, "failed {idx}");
    }
}

const WRAP: &str = "plain _italics
italics_";

#[test]
fn test_wrap_pattern() {
    let expected = vec![
        MdTree::PlainText("plain "),
        MdTree::Emphasis("italics"),
        MdTree::Emphasis(" "),
        MdTree::Emphasis("italics"),
    ]
    .into();
    let res = entrypoint(WRAP);
    assert_eq!(res, expected);
}

const WRAP_NOTXT: &str = r"_italics_
**bold**";

#[test]
fn test_wrap_notxt() {
    let expected =
        vec![MdTree::Emphasis("italics"), MdTree::PlainText(" "), MdTree::Strong("bold")].into();
    let res = entrypoint(WRAP_NOTXT);
    assert_eq!(res, expected);
}

const MIXED_LIST: &str = r"start
- _italics item_
<!-- comment -->
- **bold item**
  second line [link1](foobar1)
  third line [link2][link-foo]
-   :crab:
    extra indent
end
[link-foo]: foobar2
";

#[test]
fn test_list() {
    let expected = vec![
        MdTree::PlainText("start"),
        MdTree::ParagraphBreak,
        MdTree::UnorderedListItem(vec![MdTree::Emphasis("italics item")].into()),
        MdTree::LineBreak,
        MdTree::UnorderedListItem(
            vec![
                MdTree::Strong("bold item"),
                MdTree::PlainText(" second line "),
                MdTree::Link { disp: "link1", link: "foobar1" },
                MdTree::PlainText(" third line "),
                MdTree::Link { disp: "link2", link: "foobar2" },
            ]
            .into(),
        ),
        MdTree::LineBreak,
        MdTree::UnorderedListItem(
            vec![MdTree::PlainText("🦀"), MdTree::PlainText(" extra indent")].into(),
        ),
        MdTree::ParagraphBreak,
        MdTree::PlainText("end"),
    ]
    .into();
    let res = entrypoint(MIXED_LIST);
    assert_eq!(res, expected);
}

const SMOOSHED: &str = r#"
start
### heading
1. ordered item
```rust
println!("Hello, world!");
```
`inline`
``end``
"#;

#[test]
fn test_without_breaks() {
    let expected = vec![
        MdTree::PlainText("start"),
        MdTree::ParagraphBreak,
        MdTree::Heading(3, vec![MdTree::PlainText("heading")].into()),
        MdTree::OrderedListItem(1, vec![MdTree::PlainText("ordered item")].into()),
        MdTree::ParagraphBreak,
        MdTree::CodeBlock { txt: r#"println!("Hello, world!");"#, lang: Some("rust") },
        MdTree::ParagraphBreak,
        MdTree::CodeInline("inline"),
        MdTree::PlainText(" "),
        MdTree::CodeInline("end"),
    ]
    .into();
    let res = entrypoint(SMOOSHED);
    assert_eq!(res, expected);
}

const CODE_STARTLINE: &str = r#"
start
`code`
middle
`more code`
end
"#;

#[test]
fn test_code_at_start() {
    let expected = vec![
        MdTree::PlainText("start"),
        MdTree::PlainText(" "),
        MdTree::CodeInline("code"),
        MdTree::PlainText(" "),
        MdTree::PlainText("middle"),
        MdTree::PlainText(" "),
        MdTree::CodeInline("more code"),
        MdTree::PlainText(" "),
        MdTree::PlainText("end"),
    ]
    .into();
    let res = entrypoint(CODE_STARTLINE);
    assert_eq!(res, expected);
}

#[test]
fn test_code_in_parens() {
    let expected =
        vec![MdTree::PlainText("("), MdTree::CodeInline("Foo"), MdTree::PlainText(")")].into();
    let res = entrypoint("(`Foo`)");
    assert_eq!(res, expected);
}

const LIST_WITH_SPACE: &str = "
para
 * l1
 * l2
";

#[test]
fn test_list_with_space() {
    let expected = vec![
        MdTree::PlainText("para"),
        MdTree::ParagraphBreak,
        MdTree::UnorderedListItem(vec![MdTree::PlainText("l1")].into()),
        MdTree::LineBreak,
        MdTree::UnorderedListItem(vec![MdTree::PlainText("l2")].into()),
    ]
    .into();
    let res = entrypoint(LIST_WITH_SPACE);
    assert_eq!(res, expected);
}

const SNAKE_CASE: &str = "
foo*bar*
foo**bar**
foo_bar_
foo__bar__
";

#[test]
fn test_snake_case() {
    let expected = vec![
        MdTree::PlainText("foo"),
        MdTree::Emphasis("bar"),
        MdTree::PlainText(" "),
        MdTree::PlainText("foo"),
        MdTree::Strong("bar"),
        MdTree::PlainText(" "),
        MdTree::PlainText("foo_bar_"),
        MdTree::PlainText(" "),
        MdTree::PlainText("foo__bar__"),
    ]
    .into();
    let res = entrypoint(SNAKE_CASE);
    assert_eq!(res, expected);
}
