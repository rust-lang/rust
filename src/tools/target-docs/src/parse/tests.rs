use std::path::PathBuf;

fn assert_is_err_contains(pat: &str, err: Result<impl std::fmt::Debug, eyre::Report>) {
    let s = err.expect_err("expected error").to_string();
    if !s.contains(pat) {
        panic!("error did not contain '{pat}': {s}");
    }
}

#[test]
fn no_frontmatter() {
    let name = "archlinux-unknown-linux-gnu"; // arch linux is an arch, right?
    let content = "";
    assert_is_err_contains(
        "missing frontmatter",
        super::parse_file(PathBuf::from(name), name, content),
    );
}

#[test]
fn invalid_section() {
    let name = "6502-nintendo-nes";
    let content = "
---
pattern: \"6502-nintendo-nes\"
---

## Not A Real Section
";

    assert_is_err_contains(
        "is not an allowed section name",
        super::parse_file(PathBuf::from(name), name, content),
    );
}

#[test]
fn wrong_header() {
    let name = "x86_64-known-linux-gnu";
    let content = "
---
pattern: \"x86_64-known-linux-gnu\"
---

# x86_64-known-linux-gnu
";

    assert_is_err_contains(
        "the only allowed headings are `## `",
        super::parse_file(PathBuf::from(name), name, content),
    );
}

#[test]
fn parse_correctly() {
    let name = "cat-unknown-linux-gnu";
    let content = r#"
---
pattern: cat-unknown-linux-gnu
maintainers: ["who maintains the cat?"]
---
## Requirements

This target mostly just meows and doesn't do much.

## Testing

You can pet the cat and it might respond positively.

## Cross compilation

If you're on a dog system, there might be conflicts with the cat, be careful.
But it should be possible.
    "#;

    let info = super::parse_file(PathBuf::from(name), name, content).unwrap();

    assert_eq!(info.maintainers, vec!["who maintains the cat?"]);
    assert_eq!(info.pattern, name);
    assert_eq!(
        info.sections,
        vec![
            (
                "Requirements".to_owned(),
                "This target mostly just meows and doesn't do much.".to_owned(),
            ),
            (
                "Testing".to_owned(),
                "You can pet the cat and it might respond positively.".to_owned(),
            ),
            (
                "Cross compilation".to_owned(),
                "If you're on a dog system, there might be conflicts with the cat, be careful.\nBut it should be possible.".to_owned(),
            ),
        ]
    );
}

#[test]
fn backticks() {
    let name = "microservices-unknown-linux-gnu"; // microservices are my favourite architecture
    let content = r#"
---
pattern: "microservices-unknown-linux-gnu"
---

## Testing

```text
# hello world
```
    "#;

    let info = super::parse_file(PathBuf::from(name), name, content).unwrap();

    assert_eq!(info.pattern, name);
    assert_eq!(
        info.sections,
        vec![("Testing".to_owned(), "```text\n# hello world\n```".to_owned(),),]
    );
}

#[test]
fn wrong_file_name() {
    let name = "6502-nintendo-nes";
    let content = "
---
pattern: uwu
---

## Not A Real Section
";
    assert_is_err_contains(
        "target pattern does not match file name",
        super::parse_file(PathBuf::from(name), name, content),
    )
}
