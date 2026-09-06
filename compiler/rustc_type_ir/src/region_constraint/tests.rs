use super::compute_equated_region_var_replacements_from;

#[test]
fn equated_region_var_replacements_follow_transitive_region_var_chains() {
    const REVAR_1: u8 = 1;
    const REVAR_2: u8 = 2;
    const PLACEHOLDER: u8 = 3;
    // Equated with REVAR_1, but not a current-universe candidate and not a valid partner.
    const OTHER_REVAR: u8 = 4;

    let region_outlives = [
        (REVAR_1, REVAR_2),
        (REVAR_2, REVAR_1),
        (REVAR_2, PLACEHOLDER),
        (PLACEHOLDER, REVAR_2),
        (REVAR_1, OTHER_REVAR),
        (OTHER_REVAR, REVAR_1),
    ];

    let replacements = compute_equated_region_var_replacements_from(
        &region_outlives,
        |r| matches!(r, REVAR_1 | REVAR_2),
        |r| matches!(r, REVAR_1 | REVAR_2 | OTHER_REVAR),
    );

    assert_eq!(replacements.len(), 2);
    assert_eq!(replacements.get(&REVAR_1), Some(&PLACEHOLDER));
    assert_eq!(replacements.get(&REVAR_2), Some(&PLACEHOLDER));
}
