// This file is managed by `cargo dev rename_lint` and `cargo dev deprecate_lint`.
// Prefer to use those when possible.

macro_rules! declare_with_version {
    ($name:ident($name_version:ident): &[$ty:ty] = &[$(
        #[clippy::version = $version:literal]
        $e:expr,
    )*]) => {
        pub static $name: &[$ty] = &[$($e),*];
        #[allow(unused)]
        pub static $name_version: &[&str] = &[$($version),*];
    };
}

#[rustfmt::skip]
declare_with_version! { DEPRECATED(DEPRECATED_VERSION): &[(&str, &str)] = &[
    #[clippy::version = "pre 1.29.0"]
    ("clippy::should_assert_eq", "`assert!(a == b)` can now print the values the same way `assert_eq!(a, b) can"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::extend_from_slice", "`Vec::extend_from_slice` is no longer faster than `Vec::extend` due to specialization"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::range_step_by_zero", "`Iterator::step_by(0)` now panics and is no longer an infinite iterator"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::unstable_as_slice", "`Vec::as_slice` is now stable"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::unstable_as_mut_slice", "`Vec::as_mut_slice` is now stable"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::misaligned_transmute", "split into `clippy::cast_ptr_alignment` and `clippy::transmute_ptr_to_ptr`"),
    #[clippy::version = "1.30.0"]
    ("clippy::assign_ops", "compound operators are harmless and linting on them is not in scope for clippy"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::unsafe_vector_initialization", "the suggested alternative could be substantially slower"),
    #[clippy::version = "1.39.0"]
    ("clippy::unused_collect", "`Iterator::collect` is now marked as `#[must_use]`"),
    #[clippy::version = "1.44.0"]
    ("clippy::replace_consts", "`min_value` and `max_value` are now deprecated"),
    #[clippy::version = "1.47.0"]
    ("clippy::regex_macro", "the `regex!` macro was removed from the regex crate in 2018"),
    #[clippy::version = "1.54.0"]
    ("clippy::pub_enum_variant_names", "`clippy::enum_variant_names` now covers this case via the `avoid-breaking-exported-api` config"),
    #[clippy::version = "1.54.0"]
    ("clippy::wrong_pub_self_convention", "`clippy::wrong_self_convention` now covers this case via the `avoid-breaking-exported-api` config"),
    #[clippy::version = "1.86.0"]
    ("clippy::option_map_or_err_ok", "`clippy::manual_ok_or` covers this case"),
    #[clippy::version = "1.86.0"]
    ("clippy::match_on_vec_items", "`clippy::indexing_slicing` covers indexing and slicing on `Vec<_>`"),
    // end deprecated lints. used by `cargo dev deprecate_lint`
]}

#[rustfmt::skip]
declare_with_version! { RENAMED(RENAMED_VERSION): &[(&str, &str)] = &[
    #[clippy::version = ""]
    ("clippy::almost_complete_letter_range", "clippy::almost_complete_range"),
    #[clippy::version = ""]
    ("clippy::blacklisted_name", "clippy::disallowed_names"),
    #[clippy::version = ""]
    ("clippy::block_in_if_condition_expr", "clippy::blocks_in_conditions"),
    #[clippy::version = ""]
    ("clippy::block_in_if_condition_stmt", "clippy::blocks_in_conditions"),
    #[clippy::version = ""]
    ("clippy::blocks_in_if_conditions", "clippy::blocks_in_conditions"),
    #[clippy::version = ""]
    ("clippy::box_vec", "clippy::box_collection"),
    #[clippy::version = ""]
    ("clippy::const_static_lifetime", "clippy::redundant_static_lifetimes"),
    #[clippy::version = ""]
    ("clippy::cyclomatic_complexity", "clippy::cognitive_complexity"),
    #[clippy::version = ""]
    ("clippy::derive_hash_xor_eq", "clippy::derived_hash_with_manual_eq"),
    #[clippy::version = ""]
    ("clippy::disallowed_method", "clippy::disallowed_methods"),
    #[clippy::version = ""]
    ("clippy::disallowed_type", "clippy::disallowed_types"),
    #[clippy::version = ""]
    ("clippy::eval_order_dependence", "clippy::mixed_read_write_in_expression"),
    #[clippy::version = "1.51.0"]
    ("clippy::find_map", "clippy::manual_find_map"),
    #[clippy::version = "1.53.0"]
    ("clippy::filter_map", "clippy::manual_filter_map"),
    #[clippy::version = ""]
    ("clippy::fn_address_comparisons", "unpredictable_function_pointer_comparisons"),
    #[clippy::version = ""]
    ("clippy::identity_conversion", "clippy::useless_conversion"),
    #[clippy::version = "pre 1.29.0"]
    ("clippy::if_let_redundant_pattern_matching", "clippy::redundant_pattern_matching"),
    #[clippy::version = ""]
    ("clippy::if_let_some_result", "clippy::match_result_ok"),
    #[clippy::version = ""]
    ("clippy::incorrect_clone_impl_on_copy_type", "clippy::non_canonical_clone_impl"),
    #[clippy::version = ""]
    ("clippy::incorrect_partial_ord_impl_on_ord_type", "clippy::non_canonical_partial_ord_impl"),
    #[clippy::version = ""]
    ("clippy::integer_arithmetic", "clippy::arithmetic_side_effects"),
    #[clippy::version = ""]
    ("clippy::logic_bug", "clippy::overly_complex_bool_expr"),
    #[clippy::version = ""]
    ("clippy::new_without_default_derive", "clippy::new_without_default"),
    #[clippy::version = ""]
    ("clippy::option_and_then_some", "clippy::bind_instead_of_map"),
    #[clippy::version = ""]
    ("clippy::option_expect_used", "clippy::expect_used"),
    #[clippy::version = ""]
    ("clippy::option_map_unwrap_or", "clippy::map_unwrap_or"),
    #[clippy::version = ""]
    ("clippy::option_map_unwrap_or_else", "clippy::map_unwrap_or"),
    #[clippy::version = ""]
    ("clippy::option_unwrap_used", "clippy::unwrap_used"),
    #[clippy::version = ""]
    ("clippy::overflow_check_conditional", "clippy::panicking_overflow_checks"),
    #[clippy::version = ""]
    ("clippy::ref_in_deref", "clippy::needless_borrow"),
    #[clippy::version = ""]
    ("clippy::result_expect_used", "clippy::expect_used"),
    #[clippy::version = ""]
    ("clippy::result_map_unwrap_or_else", "clippy::map_unwrap_or"),
    #[clippy::version = ""]
    ("clippy::result_unwrap_used", "clippy::unwrap_used"),
    #[clippy::version = ""]
    ("clippy::single_char_push_str", "clippy::single_char_add_str"),
    #[clippy::version = ""]
    ("clippy::stutter", "clippy::module_name_repetitions"),
    #[clippy::version = ""]
    ("clippy::thread_local_initializer_can_be_made_const", "clippy::missing_const_for_thread_local"),
    #[clippy::version = ""]
    ("clippy::to_string_in_display", "clippy::recursive_format_impl"),
    #[clippy::version = ""]
    ("clippy::unwrap_or_else_default", "clippy::unwrap_or_default"),
    #[clippy::version = ""]
    ("clippy::zero_width_space", "clippy::invisible_characters"),
    #[clippy::version = ""]
    ("clippy::cast_ref_to_mut", "invalid_reference_casting"),
    #[clippy::version = ""]
    ("clippy::clone_double_ref", "suspicious_double_ref_op"),
    #[clippy::version = ""]
    ("clippy::cmp_nan", "invalid_nan_comparisons"),
    #[clippy::version = "CURRENT_RUSTC_VERSION"]
    ("clippy::invalid_null_ptr_usage", "invalid_null_arguments"),
    #[clippy::version = "1.86.0"]
    ("clippy::double_neg", "double_negations"),
    #[clippy::version = ""]
    ("clippy::drop_bounds", "drop_bounds"),
    #[clippy::version = ""]
    ("clippy::drop_copy", "dropping_copy_types"),
    #[clippy::version = ""]
    ("clippy::drop_ref", "dropping_references"),
    #[clippy::version = ""]
    ("clippy::fn_null_check", "useless_ptr_null_checks"),
    #[clippy::version = ""]
    ("clippy::for_loop_over_option", "for_loops_over_fallibles"),
    #[clippy::version = ""]
    ("clippy::for_loop_over_result", "for_loops_over_fallibles"),
    #[clippy::version = ""]
    ("clippy::for_loops_over_fallibles", "for_loops_over_fallibles"),
    #[clippy::version = ""]
    ("clippy::forget_copy", "forgetting_copy_types"),
    #[clippy::version = ""]
    ("clippy::forget_ref", "forgetting_references"),
    #[clippy::version = ""]
    ("clippy::into_iter_on_array", "array_into_iter"),
    #[clippy::version = ""]
    ("clippy::invalid_atomic_ordering", "invalid_atomic_ordering"),
    #[clippy::version = ""]
    ("clippy::invalid_ref", "invalid_value"),
    #[clippy::version = ""]
    ("clippy::invalid_utf8_in_unchecked", "invalid_from_utf8_unchecked"),
    #[clippy::version = ""]
    ("clippy::let_underscore_drop", "let_underscore_drop"),
    #[clippy::version = "1.80.0"]
    ("clippy::maybe_misused_cfg", "unexpected_cfgs"),
    #[clippy::version = ""]
    ("clippy::mem_discriminant_non_enum", "enum_intrinsics_non_enums"),
    #[clippy::version = "1.80.0"]
    ("clippy::mismatched_target_os", "unexpected_cfgs"),
    #[clippy::version = ""]
    ("clippy::panic_params", "non_fmt_panics"),
    #[clippy::version = ""]
    ("clippy::positional_named_format_parameters", "named_arguments_used_positionally"),
    #[clippy::version = ""]
    ("clippy::temporary_cstring_as_ptr", "dangling_pointers_from_temporaries"),
    #[clippy::version = ""]
    ("clippy::undropped_manually_drops", "undropped_manually_drops"),
    #[clippy::version = ""]
    ("clippy::unknown_clippy_lints", "unknown_lints"),
    #[clippy::version = ""]
    ("clippy::unused_label", "unused_labels"),
    #[clippy::version = ""]
    ("clippy::vtable_address_comparisons", "ambiguous_wide_pointer_comparisons"),
    #[clippy::version = ""]
    ("clippy::reverse_range_loop", "clippy::reversed_empty_ranges"),
    // end renamed lints. used by `cargo dev rename_lint`
]}
