pub(crate) fn mark_fenced_blocks_as_rust(src: &str) -> String {
    let mut processed_lines = Vec::new();
    let mut in_code_block = false;
    for line in src.lines() {
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
    fn test_codeblock_adds_rust() {
        let comment = "```\nfn some_rust() {}\n```";
        assert_eq!(mark_fenced_blocks_as_rust(comment), "```rust\nfn some_rust() {}\n```");
    }
}
