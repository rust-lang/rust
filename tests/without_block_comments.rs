extern crate clippy_lints;
use clippy_lints::utils::without_block_comments;

#[test]
fn test_lines_without_block_comments() {
    let result = without_block_comments(vec!["/*", "", "*/"]);
    println!("result: {:?}", result);
    assert!(result.is_empty());

    let result = without_block_comments(vec!["", "/*", "", "*/", "#[crate_type = \"lib\"]", "/*", "", "*/", ""]);
    assert_eq!(result, vec!["", "#[crate_type = \"lib\"]", ""]);

    let result = without_block_comments(vec!["/* rust", "", "*/"]);
    assert!(result.is_empty());

    let result = without_block_comments(vec!["/* one-line comment */"]);
    assert!(result.is_empty());

    let result = without_block_comments(vec!["/* nested", "/* multi-line", "comment", "*/", "test", "*/"]);
    assert!(result.is_empty());

    let result = without_block_comments(vec!["/* nested /* inline /* comment */ test */ */"]);
    assert!(result.is_empty());

    let result = without_block_comments(vec!["foo", "bar", "baz"]);
    assert_eq!(result, vec!["foo", "bar", "baz"]);
}
