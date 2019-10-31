//! See test_utils/src/marks.rs

test_utils::marks!(
    bogus_paths
    // FIXME: restore this mark once hir is split
    name_res_works_for_broken_modules
    can_import_enum_variant
    type_var_cycles_resolve_completely
    type_var_cycles_resolve_as_possible
    type_var_resolves_to_int_var
    glob_enum
    glob_across_crates
    std_prelude
    match_ergonomics_ref
    infer_while_let
    macro_rules_from_other_crates_are_visible_with_macro_use
    prelude_is_macro_use
    coerce_merge_fail_fallback
    macro_dollar_crate_self
    macro_dollar_crate_other
);
