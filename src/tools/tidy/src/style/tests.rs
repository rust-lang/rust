use super::*;

#[test]
fn test_generate_problematic_strings() {
    let problematic_regex = RegexSet::new(
        generate_problematic_strings(
            ROOT_PROBLEMATIC_CONSTS,
            &[('A', '4'), ('B', '8'), ('E', '3'), ('0', 'F')].iter().cloned().collect(), // use "futile" F intentionally
        )
        .as_slice(),
    )
    .unwrap();
    assert!(problematic_regex.is_match("786357")); // check with no "decimal" hex digits - converted to integer
    assert!(problematic_regex.is_match("589701")); // check with "decimal" replacements - converted to integer
    assert!(problematic_regex.is_match("8FF85")); // check for hex display
    assert!(!problematic_regex.is_match("1193046")); // check for non-matching value
}
