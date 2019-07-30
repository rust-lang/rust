pub(crate) fn format_docs(src: &str) -> String {
    let mut processed_lines = Vec::new();
    let mut in_code_block = false;
    for line in src.lines() {
        if in_code_block && line.trim_start().starts_with("# ") {
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
        let comment = "```rust\n # skip1\n# skip2\n#stay1\nstay2\n```";
        assert_eq!(format_docs(comment), "```rust\n#stay1\nstay2\n```");
    }

    #[test]
    fn test_format_docs_keeps_comments_outside_of_rust_block() {
        let comment = " # stay1\n# stay2\n#stay3\nstay4";
        assert_eq!(format_docs(comment), comment);
    }
}
