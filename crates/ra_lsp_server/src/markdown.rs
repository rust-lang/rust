use ra_ide_api::Documentation;

pub(crate) fn sanitize_markdown(docs: Documentation) -> Documentation {
    let docs: String = docs.into();

    // Massage markdown
    let mut processed_lines = Vec::new();
    let mut in_code_block = false;
    for line in docs.lines() {
        if line.starts_with("```") {
            in_code_block = !in_code_block;
        }

        let line = if in_code_block && line.starts_with("```") && !line.contains("rust") {
            "```rust".into()
        } else {
            line.to_string()
        };

        processed_lines.push(line);
    }

    Documentation::new(&processed_lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codeblock_adds_rust() {
        let comment = "```\nfn some_rust() {}\n```";
        assert_eq!(
            sanitize_markdown(Documentation::new(comment)).contents(),
            "```rust\nfn some_rust() {}\n```"
        );
    }
}
