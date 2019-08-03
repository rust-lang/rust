pub(crate) fn format_docs(src: &str) -> String {
    let mut processed_lines = Vec::new();
    let mut in_code_block = false;
    for line in src.lines() {
        if in_code_block && code_line_ignored_by_rustdoc(line) {
            continue;
        }

        if line.starts_with("```") {
            in_code_block ^= true
        }

        let line = if in_code_block && line.starts_with("```") && !line.contains("rust") {
            "```rust"
        } else {
            line
        };

        processed_lines.push(line);
    }
    processed_lines.join("\n")
}

fn code_line_ignored_by_rustdoc(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed == "#" || trimmed.starts_with("# ") || trimmed.starts_with("#\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_docs_adds_rust() {
        let comment = "```\nfn some_rust() {}\n```";
        assert_eq!(format_docs(comment), "```rust\nfn some_rust() {}\n```");
    }

    #[test]
    fn test_format_docs_skips_comments_in_rust_block() {
        let comment =
            "```rust\n # skip1\n# skip2\n#stay1\nstay2\n#\n #\n   #    \n #\tskip3\n\t#\t\n```";
        assert_eq!(format_docs(comment), "```rust\n#stay1\nstay2\n```");
    }

    #[test]
    fn test_format_docs_keeps_comments_outside_of_rust_block() {
        let comment = " # stay1\n# stay2\n#stay3\nstay4\n#\n #\n   #    \n #\tstay5\n\t#\t";
        assert_eq!(format_docs(comment), comment);
    }

    #[test]
    fn test_format_docs_preserves_newlines() {
        let comment = "this\nis\nultiline";
        assert_eq!(format_docs(comment), comment);
    }

    #[test]
    fn test_code_blocks_in_comments_marked_as_rust() {
        let comment = r#"```rust
fn main(){}
```
Some comment.
```
let a = 1;
```"#;

        assert_eq!(
            format_docs(comment),
            "```rust\nfn main(){}\n```\nSome comment.\n```rust\nlet a = 1;\n```"
        );
    }

}
