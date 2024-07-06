use super::*;

#[test]
fn test_contains_problematic_const() {
    assert!(contains_problematic_const("786357")); // check with no "decimal" hex digits - converted to integer
    assert!(contains_problematic_const("589701")); // check with "decimal" replacements - converted to integer
    assert!(contains_problematic_const("8FF85")); // check for hex display
    assert!(contains_problematic_const("8fF85")); // check for case-alternating hex display
    assert!(!contains_problematic_const("1193046")); // check for non-matching value
}
