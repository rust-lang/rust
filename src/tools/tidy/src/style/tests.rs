use super::*;

#[test]
fn test_contains_problematic_const() {
    assert!(contains_problematic_const("721077")); // check with no "decimal" hex digits - converted to integer
    assert!(contains_problematic_const("524421")); // check with "decimal" replacements - converted to integer
    assert!(contains_problematic_const(&(285 * 281).to_string())); // check for hex display
    assert!(contains_problematic_const(&format!("{:x}B5", 2816))); // check for case-alternating hex display
    assert!(!contains_problematic_const("1193046")); // check for non-matching value
}
