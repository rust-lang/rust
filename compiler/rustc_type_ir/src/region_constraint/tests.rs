use super::compute_equated_region_var_replacements_from;

#[test]
fn equated_region_var_replacements_follow_transitive_region_var_chains() {
    const REVAR_1: u8 = 1;
    const REVAR_2: u8 = 2;
    const PLACEHOLDER: u8 = 3;

    let region_outlives =
        [(REVAR_1, REVAR_2), (REVAR_2, REVAR_1), (REVAR_2, PLACEHOLDER), (PLACEHOLDER, REVAR_2)];

    let replacements = compute_equated_region_var_replacements_from(
        &region_outlives,
        |r| matches!(r, REVAR_1 | REVAR_2),
        |r| matches!(r, REVAR_1 | REVAR_2),
    );

    assert_eq!(replacements, vec![(REVAR_1, PLACEHOLDER), (REVAR_2, PLACEHOLDER)]);
}
