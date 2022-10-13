lint_array_into_iter =
    this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to <{$target} as IntoIterator>::into_iter in Rust 2021
    .use_iter_suggestion = use `.iter()` instead of `.into_iter()` to avoid ambiguity
    .remove_into_iter_suggestion = or remove `.into_iter()` to iterate by value
    .use_explicit_into_iter_suggestion =
        or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value

lint_enum_intrinsics_mem_discriminant =
    the return value of `mem::discriminant` is unspecified when called with a non-enum type
    .note = the argument to `discriminant` should be a reference to an enum, but it was passed a reference to a `{$ty_param}`, which is not an enum.

lint_enum_intrinsics_mem_variant =
    the return value of `mem::variant_count` is unspecified when called with a non-enum type
    .note = the type parameter of `variant_count` should be an enum, but it was instantiated with the type `{$ty_param}`, which is not an enum.

lint_expectation = this lint expectation is unfulfilled
    .note = the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message
    .rationale = {$rationale}

lint_for_loops_over_fallibles =
    for loop over {$article} `{$ty}`. This is more readably written as an `if let` statement
    .suggestion = consider using `if let` to clear intent
    .remove_next = to iterate over `{$recv_snip}` remove the call to `next`
    .use_while_let = to check pattern in a loop use `while let`
    .use_question_mark = consider unwrapping the `Result` with `?` to iterate over its contents

lint_non_binding_let_on_sync_lock =
    non-binding let on a synchronization lock

lint_non_binding_let_on_drop_type =
    non-binding let on a type that implements `Drop`

lint_non_binding_let_suggestion =
    consider binding to an unused variable to avoid immediately dropping the value

lint_non_binding_let_multi_suggestion =
    consider immediately dropping the value

lint_deprecated_lint_name =
    lint name `{$name}` is deprecated and may not have an effect in the future.
    .suggestion = change it to

lint_renamed_or_removed_lint = {$msg}
    .suggestion = use the new name

lint_unknown_lint =
    unknown lint: `{$name}`
    .suggestion = did you mean

lint_ignored_unless_crate_specified = {$level}({$name}) is ignored unless specified at crate level

lint_unknown_gated_lint =
    unknown lint: `{$name}`
    .note = the `{$name}` lint is unstable

lint_hidden_unicode_codepoints = unicode codepoint changing visible direction of text present in {$label}
    .label = this {$label} contains {$count ->
        [one] an invisible
        *[other] invisible
    } unicode text flow control {$count ->
        [one] codepoint
        *[other] codepoints
    }
    .note = these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen
    .suggestion_remove = if their presence wasn't intentional, you can remove them
    .suggestion_escape = if you want to keep them but make them visible in your source code, you can escape them
    .no_suggestion_note_escape = if you want to keep them but make them visible in your source code, you can escape them: {$escaped}

lint_default_hash_types = prefer `{$preferred}` over `{$used}`, it has better performance
    .note = a `use rustc_data_structures::fx::{$preferred}` may be necessary

lint_query_instability = using `{$query}` can result in unstable query results
    .note = if you believe this case to be fine, allow this lint and add a comment explaining your rationale

lint_tykind_kind = usage of `ty::TyKind::<kind>`
    .suggestion = try using `ty::<kind>` directly

lint_tykind = usage of `ty::TyKind`
    .help = try using `Ty` instead

lint_ty_qualified = usage of qualified `ty::{$ty}`
    .suggestion = try importing it and using it unqualified

lint_lintpass_by_hand = implementing `LintPass` by hand
    .help = try using `declare_lint_pass!` or `impl_lint_pass!` instead

lint_non_existant_doc_keyword = found non-existing keyword `{$keyword}` used in `#[doc(keyword = \"...\")]`
    .help = only existing keywords are allowed in core/std

lint_diag_out_of_impl =
    diagnostics should only be created in `IntoDiagnostic`/`AddToDiagnostic` impls

lint_untranslatable_diag = diagnostics should be created using translatable messages

lint_bad_opt_access = {$msg}

lint_cstring_ptr = getting the inner pointer of a temporary `CString`
    .as_ptr_label = this pointer will be invalid
    .unwrap_label = this `CString` is deallocated at the end of the statement, bind it to a variable to extend its lifetime
    .note = pointers do not have a lifetime; when calling `as_ptr` the `CString` will be deallocated at the end of the statement because nothing is referencing it as far as the type system is concerned
    .help = for more information, see https://doc.rust-lang.org/reference/destructors.html

lint_multple_supertrait_upcastable = `{$ident}` is object-safe and has multiple supertraits

lint_identifier_non_ascii_char = identifier contains non-ASCII characters

lint_identifier_uncommon_codepoints = identifier contains uncommon Unicode codepoints

lint_confusable_identifier_pair = identifier pair considered confusable between `{$existing_sym}` and `{$sym}`
    .label = this is where the previous identifier occurred

lint_mixed_script_confusables =
    the usage of Script Group `{$set}` in this crate consists solely of mixed script confusables
    .includes_note = the usage includes {$includes}
    .note = please recheck to make sure their usages are indeed what you want

lint_non_fmt_panic = panic message is not a string literal
    .note = this usage of `{$name}!()` is deprecated; it will be a hard error in Rust 2021
    .more_info_note = for more information, see <https://doc.rust-lang.org/nightly/edition-guide/rust-2021/panic-macro-consistency.html>
    .supports_fmt_note = the `{$name}!()` macro supports formatting, so there's no need for the `format!()` macro here
    .supports_fmt_suggestion = remove the `format!(..)` macro call
    .display_suggestion = add a "{"{"}{"}"}" format string to `Display` the message
    .debug_suggestion =
        add a "{"{"}:?{"}"}" format string to use the `Debug` implementation of `{$ty}`
    .panic_suggestion = {$already_suggested ->
        [true] or use
        *[false] use
    } std::panic::panic_any instead

lint_non_fmt_panic_unused =
    panic message contains {$count ->
        [one] an unused
        *[other] unused
    } formatting {$count ->
        [one] placeholder
        *[other] placeholders
    }
    .note = this message is not used as a format string when given without arguments, but will be in Rust 2021
    .add_args_suggestion = add the missing {$count ->
        [one] argument
        *[other] arguments
    }
    .add_fmt_suggestion = or add a "{"{"}{"}"}" format string to use the message literally

lint_non_fmt_panic_braces =
    panic message contains {$count ->
        [one] a brace
        *[other] braces
    }
    .note = this message is not used as a format string, but will be in Rust 2021
    .suggestion = add a "{"{"}{"}"}" format string to use the message literally

lint_non_camel_case_type = {$sort} `{$name}` should have an upper camel case name
    .suggestion = convert the identifier to upper camel case
    .label = should have an UpperCamelCase name

lint_non_snake_case = {$sort} `{$name}` should have a snake case name
    .rename_or_convert_suggestion = rename the identifier or convert it to a snake case raw identifier
    .cannot_convert_note = `{$sc}` cannot be used as a raw identifier
    .rename_suggestion = rename the identifier
    .convert_suggestion = convert the identifier to snake case
    .help = convert the identifier to snake case: `{$sc}`
    .label = should have a snake_case name

lint_non_upper_case_global = {$sort} `{$name}` should have an upper case name
    .suggestion = convert the identifier to upper case
    .label = should have an UPPER_CASE name

lint_noop_method_call = call to `.{$method}()` on a reference in this situation does nothing
    .label = unnecessary method call
    .note = the type `{$receiver_ty}` which `{$method}` is being called on is the same as the type returned from `{$method}`, so the method call does not do anything and can be removed

lint_pass_by_value = passing `{$ty}` by reference
    .suggestion = try passing by value

lint_redundant_semicolons =
    unnecessary trailing {$multiple ->
        [true] semicolons
        *[false] semicolon
    }
    .suggestion = remove {$multiple ->
        [true] these semicolons
        *[false] this semicolon
    }

lint_drop_trait_constraints =
    bounds on `{$predicate}` are most likely incorrect, consider instead using `{$needs_drop}` to detect whether a type can be trivially dropped

lint_drop_glue =
    types that do not implement `Drop` can still have drop glue, consider instead using `{$needs_drop}` to detect whether a type is trivially dropped

lint_range_endpoint_out_of_range = range endpoint is out of range for `{$ty}`
    .suggestion = use an inclusive range instead

lint_overflowing_bin_hex = literal out of range for `{$ty}`
    .negative_note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}`
    .negative_becomes_note = and the value `-{$lit}` will become `{$actually}{$ty}`
    .positive_note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}` and will become `{$actually}{$ty}`
    .suggestion = consider using the type `{$suggestion_ty}` instead
    .help = consider using the type `{$suggestion_ty}` instead

lint_overflowing_int = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`
    .help = consider using the type `{$suggestion_ty}` instead

lint_only_cast_u8_to_char = only `u8` can be cast into `char`
    .suggestion = use a `char` literal instead

lint_overflowing_uint = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`

lint_overflowing_literal = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` and will be converted to `{$ty}::INFINITY`

lint_unused_comparisons = comparison is useless due to type limits

lint_improper_ctypes = `extern` {$desc} uses type `{$ty}`, which is not FFI-safe
    .label = not FFI-safe
    .note = the type is defined here

lint_improper_ctypes_opaque = opaque types have no C equivalent

lint_improper_ctypes_fnptr_reason = this function pointer has Rust-specific calling convention
lint_improper_ctypes_fnptr_help = consider using an `extern fn(...) -> ...` function pointer instead

lint_improper_ctypes_tuple_reason = tuples have unspecified layout
lint_improper_ctypes_tuple_help = consider using a struct instead

lint_improper_ctypes_str_reason = string slices have no C equivalent
lint_improper_ctypes_str_help = consider using `*const u8` and a length instead

lint_improper_ctypes_dyn = trait objects have no C equivalent

lint_improper_ctypes_slice_reason = slices have no C equivalent
lint_improper_ctypes_slice_help = consider using a raw pointer instead

lint_improper_ctypes_128bit = 128-bit integers don't currently have a known stable ABI

lint_improper_ctypes_char_reason = the `char` type has no C equivalent
lint_improper_ctypes_char_help = consider using `u32` or `libc::wchar_t` instead

lint_improper_ctypes_non_exhaustive = this enum is non-exhaustive
lint_improper_ctypes_non_exhaustive_variant = this enum has non-exhaustive variants

lint_improper_ctypes_enum_repr_reason = enum has no representation hint
lint_improper_ctypes_enum_repr_help =
    consider adding a `#[repr(C)]`, `#[repr(transparent)]`, or integer `#[repr(...)]` attribute to this enum

lint_improper_ctypes_struct_fieldless_reason = this struct has no fields
lint_improper_ctypes_struct_fieldless_help = consider adding a member to this struct

lint_improper_ctypes_union_fieldless_reason = this union has no fields
lint_improper_ctypes_union_fieldless_help = consider adding a member to this union

lint_improper_ctypes_struct_non_exhaustive = this struct is non-exhaustive
lint_improper_ctypes_union_non_exhaustive = this union is non-exhaustive

lint_improper_ctypes_struct_layout_reason = this struct has unspecified layout
lint_improper_ctypes_struct_layout_help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this struct

lint_improper_ctypes_union_layout_reason = this union has unspecified layout
lint_improper_ctypes_union_layout_help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this union

lint_improper_ctypes_box = box cannot be represented as a single pointer

lint_improper_ctypes_enum_phantomdata = this enum contains a PhantomData field

lint_improper_ctypes_struct_zst = this struct contains only zero-sized fields

lint_improper_ctypes_array_reason = passing raw arrays by value is not FFI-safe
lint_improper_ctypes_array_help = consider passing a pointer to the array

lint_improper_ctypes_only_phantomdata = composed only of `PhantomData`

lint_variant_size_differences =
    enum variant is more than three times larger ({$largest} bytes) than the next largest

lint_atomic_ordering_load = atomic loads cannot have `Release` or `AcqRel` ordering
    .help = consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`

lint_atomic_ordering_store = atomic stores cannot have `Acquire` or `AcqRel` ordering
    .help = consider using ordering modes `Release`, `SeqCst` or `Relaxed`

lint_atomic_ordering_fence = memory fences cannot have `Relaxed` ordering
    .help = consider using ordering modes `Acquire`, `Release`, `AcqRel` or `SeqCst`

lint_atomic_ordering_invalid = `{$method}`'s failure ordering may not be `Release` or `AcqRel`, since a failed `{$method}` does not result in a write
    .label = invalid failure ordering
    .help = consider using `Acquire` or `Relaxed` failure ordering instead

lint_unused_op = unused {$op} that must be used
    .label = the {$op} produces a value
    .suggestion = use `let _ = ...` to ignore the resulting value

lint_unused_result = unused result of type `{$ty}`

lint_unused_closure =
    unused {$pre}{$count ->
        [one] closure
        *[other] closures
    }{$post} that must be used
    .note = closures are lazy and do nothing unless called

lint_unused_generator =
    unused {$pre}{$count ->
        [one] generator
        *[other] generator
    }{$post} that must be used
    .note = generators are lazy and do nothing unless resumed

lint_unused_def = unused {$pre}`{$def}`{$post} that must be used
    .suggestion = use `let _ = ...` to ignore the resulting value

lint_path_statement_drop = path statement drops value
    .suggestion = use `drop` to clarify the intent

lint_path_statement_no_effect = path statement with no effect

lint_unused_delim = unnecessary {$delim} around {$item}
    .suggestion = remove these {$delim}

lint_unused_import_braces = braces around {$node} is unnecessary

lint_unused_allocation = unnecessary allocation, use `&` instead
lint_unused_allocation_mut = unnecessary allocation, use `&mut` instead

lint_builtin_while_true = denote infinite loops with `loop {"{"} ... {"}"}`
    .suggestion = use `loop`

lint_builtin_box_pointers = type uses owned (Box type) pointers: {$ty}

lint_builtin_non_shorthand_field_patterns = the `{$ident}:` in this pattern is redundant
    .suggestion = use shorthand field pattern

lint_builtin_overridden_symbol_name =
    the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them

lint_builtin_overridden_symbol_section =
    the program's behavior with overridden link sections on items is unpredictable and Rust cannot provide guarantees when you manually override them

lint_builtin_allow_internal_unsafe =
    `allow_internal_unsafe` allows defining macros using unsafe without triggering the `unsafe_code` lint at their call site

lint_builtin_unsafe_block = usage of an `unsafe` block

lint_builtin_unsafe_trait = declaration of an `unsafe` trait

lint_builtin_unsafe_impl = implementation of an `unsafe` trait

lint_builtin_no_mangle_fn = declaration of a `no_mangle` function
lint_builtin_export_name_fn = declaration of a function with `export_name`
lint_builtin_link_section_fn = declaration of a function with `link_section`

lint_builtin_no_mangle_static = declaration of a `no_mangle` static
lint_builtin_export_name_static = declaration of a static with `export_name`
lint_builtin_link_section_static = declaration of a static with `link_section`

lint_builtin_no_mangle_method = declaration of a `no_mangle` method
lint_builtin_export_name_method = declaration of a method with `export_name`

lint_builtin_decl_unsafe_fn = declaration of an `unsafe` function
lint_builtin_decl_unsafe_method = declaration of an `unsafe` method
lint_builtin_impl_unsafe_method = implementation of an `unsafe` method

lint_builtin_missing_doc = missing documentation for {$article} {$desc}

lint_builtin_missing_copy_impl = type could implement `Copy`; consider adding `impl Copy`

lint_builtin_missing_debug_impl =
    type does not implement `{$debug}`; consider adding `#[derive(Debug)]` or a manual implementation

lint_builtin_anonymous_params = anonymous parameters are deprecated and will be removed in the next edition
    .suggestion = try naming the parameter or explicitly ignoring it

lint_builtin_deprecated_attr_link = use of deprecated attribute `{$name}`: {$reason}. See {$link}
    .msg_suggestion = {$msg}
    .default_suggestion = remove this attribute
lint_builtin_deprecated_attr_used = use of deprecated attribute `{$name}`: no longer used.
lint_builtin_deprecated_attr_default_suggestion = remove this attribute

lint_builtin_unused_doc_comment = unused doc comment
    .label = rustdoc does not generate documentation for {$kind}
    .plain_help = use `//` for a plain comment
    .block_help = use `/* */` for a plain comment

lint_builtin_no_mangle_generic = functions generic over types or consts must be mangled
    .suggestion = remove this attribute

lint_builtin_const_no_mangle = const items should never be `#[no_mangle]`
    .suggestion = try a static value

lint_builtin_mutable_transmutes =
    transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell

lint_builtin_unstable_features = unstable feature

lint_ungated_async_fn_track_caller = `#[track_caller]` on async functions is a no-op
     .label = this function will not propagate the caller location

lint_builtin_unreachable_pub = unreachable `pub` {$what}
    .suggestion = consider restricting its visibility
    .help = or consider exporting it for use by other crates

lint_builtin_unexpected_cli_config_name = unexpected `{$name}` as condition name
    .help = was set with `--cfg` but isn't in the `--check-cfg` expected names

lint_builtin_unexpected_cli_config_value = unexpected condition value `{$value}` for condition name `{$name}`
    .help = was set with `--cfg` but isn't in the `--check-cfg` expected values

lint_builtin_type_alias_bounds_help = use fully disambiguated paths (i.e., `<T as Trait>::Assoc`) to refer to associated types in type aliases

lint_builtin_type_alias_where_clause = where clauses are not enforced in type aliases
    .suggestion = the clause will not be checked when the type alias is used, and should be removed

lint_builtin_type_alias_generic_bounds = bounds on generic parameters are not enforced in type aliases
    .suggestion = the bound will not be checked when the type alias is used, and should be removed

lint_builtin_trivial_bounds = {$predicate_kind_name} bound {$predicate} does not depend on any type or lifetime parameters

lint_builtin_ellipsis_inclusive_range_patterns = `...` range patterns are deprecated
    .suggestion = use `..=` for an inclusive range

lint_builtin_unnameable_test_items = cannot test inner items

lint_builtin_keyword_idents = `{$kw}` is a keyword in the {$next} edition
    .suggestion = you can use a raw identifier to stay compatible

lint_builtin_explicit_outlives = outlives requirements can be inferred
    .suggestion = remove {$count ->
        [one] this bound
        *[other] these bounds
    }

lint_builtin_incomplete_features = the feature `{$name}` is incomplete and may not be safe to use and/or cause compiler crashes
    .note = see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information
    .help = consider using `min_{$name}` instead, which is more stable and complete

lint_builtin_unpermitted_type_init_zeroed = the type `{$ty}` does not permit zero-initialization
lint_builtin_unpermitted_type_init_unint = the type `{$ty}` does not permit being left uninitialized

lint_builtin_unpermitted_type_init_label = this code causes undefined behavior when executed
lint_builtin_unpermitted_type_init_label_suggestion = help: use `MaybeUninit<T>` instead, and only call `assume_init` after initialization is done

lint_builtin_clashing_extern_same_name = `{$this}` redeclared with a different signature
    .previous_decl_label = `{$orig}` previously declared here
    .mismatch_label = this signature doesn't match the previous declaration
lint_builtin_clashing_extern_diff_name = `{$this}` redeclares `{$orig}` with a different signature
    .previous_decl_label = `{$orig}` previously declared here
    .mismatch_label = this signature doesn't match the previous declaration

lint_builtin_deref_nullptr = dereferencing a null pointer
    .label = this code causes undefined behavior when executed

lint_builtin_asm_labels = avoid using named labels in inline assembly

lint_builtin_special_module_name_used_lib = found module declaration for lib.rs
    .note = lib.rs is the root of this crate's library target
    .help = to refer to it from other targets, use the library's name as the path

lint_builtin_special_module_name_used_main = found module declaration for main.rs
    .note = a binary crate cannot be used as library

lint_supertrait_as_deref_target = `{$t}` implements `Deref` with supertrait `{$target_principal}` as target
    .label = target type is set here

lint_overruled_attribute = {$lint_level}({$lint_source}) incompatible with previous forbid
    .label = overruled by previous forbid

lint_default_source = `forbid` lint level is the default for {$id}

lint_node_source = `forbid` level set here
    .note = {$reason}

lint_command_line_source = `forbid` lint level was set on command line

lint_malformed_attribute = malformed lint attribute input

lint_bad_attribute_argument = bad attribute argument

lint_reason_must_be_string_literal = reason must be a string literal

lint_reason_must_come_last = reason in lint attribute must come last

lint_unknown_tool_in_scoped_lint = unknown tool name `{$tool_name}` found in scoped lint: `{$tool_name}::{$lint_name}`
    .help = add `#![register_tool({$tool_name})]` to the crate root

lint_unsupported_group = `{$lint_group}` lint group is not supported with ´--force-warn´

lint_requested_level = requested on the command line with `{$level} {$lint_name}`

lint_check_name_unknown = unknown lint: `{$lint_name}`
    .help = did you mean: `{$suggestion}`

lint_check_name_unknown_tool = unknown lint tool: `{$tool_name}`

lint_check_name_warning = {$msg}

lint_check_name_deprecated = lint name `{$lint_name}` is deprecated and does not have an effect anymore. Use: {$new_name}

lint_opaque_hidden_inferred_bound = opaque type `{$ty}` does not satisfy its associated type bounds
    .specifically = this associated type bound is unsatisfied for `{$proj_ty}`

lint_opaque_hidden_inferred_bound_sugg = add this bound
