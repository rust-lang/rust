use super::*;

#[test]
fn test_generate_problematic_strings() {
    let problematic_regex = RegexSet::new(
        generate_problematic_strings(
            ROOT_PROBLEMATIC_CONSTS,
            &[('A', '4'), ('B', '8'), ('E', '3'), ('F', '0')].iter().cloned().collect(),
        )
        .as_slice(),
    )
    .unwrap();
    assert!(problematic_regex.is_match("524421")); // check for only "decimal" hex digits (converted to integer intentionally)
    assert!(problematic_regex.is_match("721077")); // check for char replacements (converted to integer intentionally)
    assert!(problematic_regex.is_match("8FF85")); // check for hex display but use "futile" F intentionally 
    assert!(!problematic_regex.is_match("1193046")); // check for non-matching value
}
