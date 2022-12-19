use super::*;

#[test]
fn test_comment() {
    const TAG: MdType = MdType::Comment;
    let pat = PATTERNS.iter().find(|p| p.tag == TAG).unwrap();
    let ctx = Context { at_line_start: true, preceded_by_break: true };

    let input = b"none<!--none-->residual";
    assert_eq!(
        pat.parse_end(input, &ctx),
        MdResult { matched: MdTree::Comment("none<!--none"), residual: b"residual" }
    );
    assert_eq!(pat.parse_start(input, &ctx), None);

    let input = b"<!--comment\n-->residual";
    let expected = MdResult { matched: MdTree::Comment("comment\n"), residual: b"residual" };
    assert_eq!(pat.parse_start(input, &ctx), Some(expected));
}

#[test]
fn test_code_block() {
    const TAG: MdType = MdType::CodeBlock;
    let pat = PATTERNS.iter().find(|p| p.tag == TAG).unwrap();
    let ctx = Context { at_line_start: true, preceded_by_break: true };

    let input = b"none\n```\nblock\n```";
    let end_expected =
        MdResult { matched: MdTree::from_type("none\n", TAG), residual: b"\nblock\n```" };
    assert_eq!(pat.parse_end(input, &ctx), end_expected);
    assert_eq!(pat.parse_start(input, &ctx), None);

    let input = b"```\nblock\nof\ncode\n```residual";
    let expected =
        MdResult { matched: MdTree::from_type("\nblock\nof\ncode\n", TAG), residual: b"residual" };
    assert_eq!(pat.parse_start(input, &ctx), Some(expected));

    let ctx = Context { at_line_start: false, preceded_by_break: true };
    assert_eq!(pat.parse_start(input, &ctx), None);
}

#[test]
fn test_headings() {
    const TAG: MdType = MdType::Heading1;
    let pat = PATTERNS.iter().find(|p| p.tag == TAG).unwrap();
    let ctx = Context { at_line_start: true, preceded_by_break: true };

    let input = b"content\nresidual";
    let end_expected = MdResult {
        // Only match if whitespace comes after
        matched: MdTree::from_type("content", TAG),
        residual: b"\nresidual",
    };
    assert_eq!(pat.parse_end(input, &ctx), end_expected);
    assert_eq!(pat.parse_start(input, &ctx), None);

    let input = b"# content\nresidual";
    let expected = MdResult { matched: MdTree::from_type("content", TAG), residual: b"\nresidual" };
    assert_eq!(pat.parse_start(input, &ctx), Some(expected));

    let ctx = Context { at_line_start: false, preceded_by_break: true };
    assert_eq!(pat.parse_start(input, &ctx), None);
}

#[test]
fn test_code_inline() {
    const TAG: MdType = MdType::CodeInline;
    let pat = PATTERNS.iter().find(|p| p.tag == TAG).unwrap();
    let ctx = Context { at_line_start: false, preceded_by_break: true };

    let input = b"none `block` residual";
    let end_expected = MdResult {
        // Only match if whitespace comes after
        matched: MdTree::from_type("none `block", TAG),
        residual: b" residual",
    };
    assert_eq!(pat.parse_end(input, &ctx), end_expected);
    assert_eq!(pat.parse_start(input, &ctx), None);

    let input = b"`block` residual";
    let expected = MdResult { matched: MdTree::from_type("block", TAG), residual: b" residual" };
    assert_eq!(pat.parse_start(input, &ctx), Some(expected));

    let ctx = Context { at_line_start: false, preceded_by_break: false };
    assert_eq!(pat.parse_start(input, &ctx), None);
}

const MD_INPUT: &str = r#"# Headding 1

Some `inline code`

```
code block here;
more code;
```

<!-- I should disappear -->
Further `inline`, some **bold**, a bit of _italics
wrapped across lines_. We can also try (`code inside parentheses`).

Let's end with a list:

- Item 1 _italics_ example
- Item 2 **bold**

## Heading 2: Other things for `code`

_start of line_
**more start of line**

```rust
try two of everything
```
<!--
  another
  comment -->
"#;

fn expected_ast() -> MdTree<'static> {
    MdTree::Root(vec![
        MdTree::Heading1(vec![MdTree::PlainText("Headding 1")]),
        MdTree::PlainText("\n\nSome "),
        MdTree::CodeInline("inline code"),
        MdTree::PlainText("\n\n"),
        MdTree::CodeBlock("\ncode block here;\nmore code;\n"),
        MdTree::PlainText("\n\n"),
        MdTree::Comment(" I should disappear "),
        MdTree::PlainText("\nFurther "),
        MdTree::CodeInline("inline"),
        MdTree::PlainText(", some "),
        MdTree::Strong("bold"),
        MdTree::PlainText(", a bit of "),
        MdTree::Emphasis("italics\nwrapped across lines"),
        MdTree::PlainText(". We can also try ("),
        MdTree::CodeInline("code inside parentheses"),
        MdTree::PlainText(").\n\nLet's end with a list:\n\n"),
        MdTree::ListItem(vec![
            MdTree::PlainText(" Item 1 "),
            MdTree::Emphasis("italics"),
            MdTree::PlainText(" example"),
        ]),
        MdTree::PlainText("\n"),
        MdTree::ListItem(vec![MdTree::PlainText(" Item 2 "), MdTree::Strong("bold")]),
        MdTree::PlainText("\n\n"),
        MdTree::Heading2(vec![
            MdTree::PlainText("Heading 2: Other things for "),
            MdTree::CodeInline("code"),
        ]),
        MdTree::PlainText("\n\n"),
        MdTree::Emphasis("start of line"),
        MdTree::PlainText("\n"),
        MdTree::Strong("more start of line"),
        MdTree::PlainText("\n\n"),
        MdTree::CodeBlock("rust\ntry two of everything\n"),
        MdTree::PlainText("\n"),
        MdTree::Comment("\n  another\n  comment "),
        MdTree::PlainText("\n"),
    ])
}

#[test]
fn test_tree() {
    let result = create_ast(MD_INPUT);
    assert_eq!(result, expected_ast());
}
