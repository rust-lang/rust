#[test]
fn no_frontmatter() {
    let name = "archlinux-unknown-linux-gnu.md"; // arch linux is an arch, right?
    let content = "";
    assert!(super::parse_file(name, content).is_err());
}

#[test]
fn invalid_section() {
    let name = "6502-nintendo-nes.md";
    let content = "
---
---

## Not A Real Section
";

    assert!(super::parse_file(name, content).is_err());
}

#[test]
fn wrong_header() {
    let name = "x86_64-known-linux-gnu.md";
    let content = "
---
---

# x86_64-known-linux-gnu
";

    assert!(super::parse_file(name, content).is_err());
}

#[test]
fn parse_correctly() {
    let name = "cat-unknown-linux-gnu.md";
    let content = r#"
---
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

    let info = super::parse_file(name, content).unwrap();

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
---

## Testing

```text
# hello world
```
    "#;

    let info = super::parse_file(name, content).unwrap();

    assert_eq!(info.pattern, name);
    assert_eq!(
        info.sections,
        vec![("Testing".to_owned(), "```text\n# hello world\n```".to_owned(),),]
    );
}
