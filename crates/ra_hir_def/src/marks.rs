//! See test_utils/src/marks.rs

test_utils::marks!(
    bogus_paths
    name_res_works_for_broken_modules
    can_import_enum_variant
    glob_enum
    glob_across_crates
    std_prelude
    macro_rules_from_other_crates_are_visible_with_macro_use
    prelude_is_macro_use
    macro_dollar_crate_self
    macro_dollar_crate_other
);
