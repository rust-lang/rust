lint_abs_path_with_module = absolute paths must start with `self`, `super`, `crate`, or an external crate name in the 2018 edition
    .suggestion = use `crate`

lint_ambiguous_glob_reexport = ambiguous glob re-exports
    .label_first_reexport = the name `{$name}` in the {$namespace} namespace is first re-exported here
    .label_duplicate_reexport = but the name `{$name}` in the {$namespace} namespace is also re-exported here

lint_ambiguous_negative_literals = `-` has lower precedence than method calls, which might be unexpected
    .example = e.g. `-4.abs()` equals `-4`; while `(-4).abs()` equals `4`
    .negative_literal = add parentheses around the `-` and the literal to call the method on a negative literal
    .current_behavior = add parentheses around the literal and the method call to keep the current behavior

lint_ambiguous_wide_pointer_comparisons = ambiguous wide pointer comparison, the comparison includes metadata which may not be expected
    .addr_metadata_suggestion = use explicit `std::ptr::eq` method to compare metadata and addresses
    .addr_suggestion = use `std::ptr::addr_eq` or untyped pointers to only compare their addresses
    .cast_suggestion = use untyped pointers to only compare their addresses
    .expect_suggestion = or expect the lint to compare the pointers metadata and addresses

lint_associated_const_elided_lifetime = {$elided ->
        [true] `&` without an explicit lifetime name cannot be used here
        *[false] `'_` cannot be used here
    }
    .suggestion = use the `'static` lifetime
    .note = cannot automatically infer `'static` because of other lifetimes in scope

lint_async_fn_in_trait = use of `async fn` in public traits is discouraged as auto trait bounds cannot be specified
    .note = you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
    .suggestion = you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change

lint_atomic_ordering_fence = memory fences cannot have `Relaxed` ordering
    .help = consider using ordering modes `Acquire`, `Release`, `AcqRel` or `SeqCst`

lint_atomic_ordering_invalid = `{$method}`'s failure ordering may not be `Release` or `AcqRel`, since a failed `{$method}` does not result in a write
    .label = invalid failure ordering
    .help = consider using `Acquire` or `Relaxed` failure ordering instead

lint_atomic_ordering_load = atomic loads cannot have `Release` or `AcqRel` ordering
    .help = consider using ordering modes `Acquire`, `SeqCst` or `Relaxed`

lint_atomic_ordering_store = atomic stores cannot have `Acquire` or `AcqRel` ordering
    .help = consider using ordering modes `Release`, `SeqCst` or `Relaxed`

lint_avoid_att_syntax =
    avoid using `.att_syntax`, prefer using `options(att_syntax)` instead

lint_avoid_intel_syntax =
    avoid using `.intel_syntax`, Intel syntax is the default

lint_bad_attribute_argument = bad attribute argument

lint_bad_opt_access = {$msg}

lint_break_with_label_and_loop = this labeled break expression is easy to confuse with an unlabeled break with a labeled value expression
    .suggestion = wrap this expression in parentheses

lint_builtin_allow_internal_unsafe =
    `allow_internal_unsafe` allows defining macros using unsafe without triggering the `unsafe_code` lint at their call site

lint_builtin_anonymous_params = anonymous parameters are deprecated and will be removed in the next edition
    .suggestion = try naming the parameter or explicitly ignoring it

lint_builtin_clashing_extern_diff_name = `{$this}` redeclares `{$orig}` with a different signature
    .previous_decl_label = `{$orig}` previously declared here
    .mismatch_label = this signature doesn't match the previous declaration

lint_builtin_clashing_extern_same_name = `{$this}` redeclared with a different signature
    .previous_decl_label = `{$orig}` previously declared here
    .mismatch_label = this signature doesn't match the previous declaration
lint_builtin_const_no_mangle = const items should never be `#[no_mangle]`
    .suggestion = try a static value

lint_builtin_decl_unsafe_fn = declaration of an `unsafe` function
lint_builtin_decl_unsafe_method = declaration of an `unsafe` method

lint_builtin_deprecated_attr_link = use of deprecated attribute `{$name}`: {$reason}. See {$link}
    .msg_suggestion = {$msg}
    .default_suggestion = remove this attribute
lint_builtin_deref_nullptr = dereferencing a null pointer
    .label = this code causes undefined behavior when executed

lint_builtin_double_negations = use of a double negation
    .note = the prefix `--` could be misinterpreted as a decrement operator which exists in other languages
    .note_decrement = use `-= 1` if you meant to decrement the value
    .add_parens_suggestion = add parentheses for clarity

lint_builtin_ellipsis_inclusive_range_patterns = `...` range patterns are deprecated
    .suggestion = use `..=` for an inclusive range

lint_builtin_explicit_outlives = outlives requirements can be inferred
    .suggestion = remove {$count ->
        [one] this bound
        *[other] these bounds
    }

lint_builtin_export_name_fn = declaration of a function with `export_name`
lint_builtin_export_name_method = declaration of a method with `export_name`
lint_builtin_export_name_static = declaration of a static with `export_name`

lint_builtin_global_asm = usage of `core::arch::global_asm`
lint_builtin_global_macro_unsafety = using this macro is unsafe even though it does not need an `unsafe` block

lint_builtin_impl_unsafe_method = implementation of an `unsafe` method

lint_builtin_incomplete_features = the feature `{$name}` is incomplete and may not be safe to use and/or cause compiler crashes
    .note = see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information
    .help = consider using `min_{$name}` instead, which is more stable and complete

lint_builtin_internal_features = the feature `{$name}` is internal to the compiler or standard library
    .note = using it is strongly discouraged

lint_builtin_keyword_idents = `{$kw}` is a keyword in the {$next} edition
    .suggestion = you can use a raw identifier to stay compatible

lint_builtin_link_section_fn = declaration of a function with `link_section`

lint_builtin_link_section_static = declaration of a static with `link_section`

lint_builtin_missing_copy_impl = type could implement `Copy`; consider adding `impl Copy`

lint_builtin_missing_debug_impl =
    type does not implement `{$debug}`; consider adding `#[derive(Debug)]` or a manual implementation

lint_builtin_missing_doc = missing documentation for {$article} {$desc}

lint_builtin_mutable_transmutes =
    transmuting &T to &mut T is undefined behavior, even if the reference is unused, consider instead using an UnsafeCell

lint_builtin_no_mangle_fn = declaration of a `no_mangle` function
lint_builtin_no_mangle_generic = functions generic over types or consts must be mangled
    .suggestion = remove this attribute

lint_builtin_no_mangle_method = declaration of a `no_mangle` method
lint_builtin_no_mangle_static = declaration of a `no_mangle` static
lint_builtin_non_shorthand_field_patterns = the `{$ident}:` in this pattern is redundant
    .suggestion = use shorthand field pattern

lint_builtin_overridden_symbol_name =
    the linker's behavior with multiple libraries exporting duplicate symbol names is undefined and Rust cannot provide guarantees when you manually override them

lint_builtin_overridden_symbol_section =
    the program's behavior with overridden link sections on items is unpredictable and Rust cannot provide guarantees when you manually override them

lint_builtin_special_module_name_used_lib = found module declaration for lib.rs
    .note = lib.rs is the root of this crate's library target
    .help = to refer to it from other targets, use the library's name as the path

lint_builtin_special_module_name_used_main = found module declaration for main.rs
    .note = a binary crate cannot be used as library

lint_builtin_trivial_bounds = {$predicate_kind_name} bound {$predicate} does not depend on any type or lifetime parameters

lint_builtin_type_alias_bounds_enable_feat_help = add `#![feature(lazy_type_alias)]` to the crate attributes to enable the desired semantics
lint_builtin_type_alias_bounds_label = will not be checked at usage sites of the type alias
lint_builtin_type_alias_bounds_limitation_note = this is a known limitation of the type checker that may be lifted in a future edition.
    see issue #112792 <https://github.com/rust-lang/rust/issues/112792> for more information
lint_builtin_type_alias_bounds_param_bounds = bounds on generic parameters in type aliases are not enforced
    .suggestion = remove {$count ->
        [one] this bound
        *[other] these bounds
    }
lint_builtin_type_alias_bounds_qualify_assoc_tys_sugg = fully qualify this associated type
lint_builtin_type_alias_bounds_where_clause = where clauses on type aliases are not enforced
    .suggestion = remove this where clause

lint_builtin_unpermitted_type_init_label = this code causes undefined behavior when executed
lint_builtin_unpermitted_type_init_label_suggestion = help: use `MaybeUninit<T>` instead, and only call `assume_init` after initialization is done

lint_builtin_unpermitted_type_init_uninit = the type `{$ty}` does not permit being left uninitialized

lint_builtin_unpermitted_type_init_zeroed = the type `{$ty}` does not permit zero-initialization
lint_builtin_unreachable_pub = unreachable `pub` {$what}
    .suggestion = consider restricting its visibility
    .help = or consider exporting it for use by other crates

lint_builtin_unsafe_block = usage of an `unsafe` block

lint_builtin_unsafe_extern_block = usage of an `unsafe extern` block

lint_builtin_unsafe_impl = implementation of an `unsafe` trait

lint_builtin_unsafe_trait = declaration of an `unsafe` trait

lint_builtin_unstable_features = use of an unstable feature

lint_builtin_unused_doc_comment = unused doc comment
    .label = rustdoc does not generate documentation for {$kind}
    .plain_help = use `//` for a plain comment
    .block_help = use `/* */` for a plain comment

lint_builtin_while_true = denote infinite loops with `loop {"{"} ... {"}"}`
    .suggestion = use `loop`

lint_byte_slice_in_packed_struct_with_derive = {$ty} slice in a packed struct that derives a built-in trait
    .help = consider implementing the trait by hand, or remove the `packed` attribute

lint_cfg_attr_no_attributes =
    `#[cfg_attr]` does not expand to any attributes

lint_check_name_unknown_tool = unknown lint tool: `{$tool_name}`

lint_closure_returning_async_block = closure returning async block can be made into an async closure
    .label = this async block can be removed, and the closure can be turned into an async closure
    .suggestion = turn this into an async closure

lint_command_line_source = `forbid` lint level was set on command line

lint_confusable_identifier_pair = found both `{$existing_sym}` and `{$sym}` as identifiers, which look alike
    .current_use = this identifier can be confused with `{$existing_sym}`
    .other_use = other identifier used here

lint_custom_inner_attribute_unstable = custom inner attributes are unstable

lint_dangling_pointers_from_temporaries = a dangling pointer will be produced because the temporary `{$ty}` will be dropped
    .label_ptr = this pointer will immediately be invalid
    .label_temporary = this `{$ty}` is deallocated at the end of the statement, bind it to a variable to extend its lifetime
    .note = pointers do not have a lifetime; when calling `{$callee}` the `{$ty}` will be deallocated at the end of the statement because nothing is referencing it as far as the type system is concerned
    .help_bind = you must make sure that the variable you bind the `{$ty}` to lives at least as long as the pointer returned by the call to `{$callee}`
    .help_returned = in particular, if this pointer is returned from the current function, binding the `{$ty}` inside the function will not suffice
    .help_visit = for more information, see <https://doc.rust-lang.org/reference/destructors.html>

lint_default_hash_types = prefer `{$preferred}` over `{$used}`, it has better performance
    .note = a `use rustc_data_structures::fx::{$preferred}` may be necessary

lint_default_source = `forbid` lint level is the default for {$id}

lint_deprecated_lint_name =
    lint name `{$name}` is deprecated and may not have an effect in the future
    .suggestion = change it to
    .help = change it to {$replace}

lint_deprecated_where_clause_location = where clause not allowed here
    .note = see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information
    .suggestion_move_to_end = move it to the end of the type declaration
    .suggestion_remove_where = remove this `where`

lint_diag_out_of_impl =
    diagnostics should only be created in `Diagnostic`/`Subdiagnostic`/`LintDiagnostic` impls

lint_drop_glue =
    types that do not implement `Drop` can still have drop glue, consider instead using `{$needs_drop}` to detect whether a type is trivially dropped

lint_drop_trait_constraints =
    bounds on `{$predicate}` are most likely incorrect, consider instead using `{$needs_drop}` to detect whether a type can be trivially dropped

lint_dropping_copy_types = calls to `std::mem::drop` with a value that implements `Copy` does nothing
    .label = argument has type `{$arg_ty}`

lint_dropping_references = calls to `std::mem::drop` with a reference instead of an owned value does nothing
    .label = argument has type `{$arg_ty}`

lint_duplicate_macro_attribute =
    duplicated attribute

lint_duplicate_matcher_binding = duplicate matcher binding

lint_enum_intrinsics_mem_discriminant =
    the return value of `mem::discriminant` is unspecified when called with a non-enum type
    .note = the argument to `discriminant` should be a reference to an enum, but it was passed a reference to a `{$ty_param}`, which is not an enum

lint_enum_intrinsics_mem_variant =
    the return value of `mem::variant_count` is unspecified when called with a non-enum type
    .note = the type parameter of `variant_count` should be an enum, but it was instantiated with the type `{$ty_param}`, which is not an enum

lint_expectation = this lint expectation is unfulfilled
    .note = the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message
    .rationale = {$rationale}

lint_extern_crate_not_idiomatic = `extern crate` is not idiomatic in the new edition
    .suggestion = convert it to a `use`

lint_extern_without_abi = `extern` declarations without an explicit ABI are deprecated
    .label = ABI should be specified here
    .suggestion = explicitly specify the {$default_abi} ABI

lint_for_loops_over_fallibles =
    for loop over {$article} `{$ref_prefix}{$ty}`. This is more readably written as an `if let` statement
    .suggestion = consider using `if let` to clear intent
    .remove_next = to iterate over `{$recv_snip}` remove the call to `next`
    .use_while_let = to check pattern in a loop use `while let`
    .use_question_mark = consider unwrapping the `Result` with `?` to iterate over its contents

lint_forgetting_copy_types = calls to `std::mem::forget` with a value that implements `Copy` does nothing
    .label = argument has type `{$arg_ty}`

lint_forgetting_references = calls to `std::mem::forget` with a reference instead of an owned value does nothing
    .label = argument has type `{$arg_ty}`

lint_hidden_glob_reexport = private item shadows public glob re-export
    .note_glob_reexport = the name `{$name}` in the {$namespace} namespace is supposed to be publicly re-exported here
    .note_private_item = but the private item here shadows it

lint_hidden_lifetime_parameters = hidden lifetime parameters in types are deprecated

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

lint_identifier_non_ascii_char = identifier contains non-ASCII characters

lint_identifier_uncommon_codepoints = identifier contains {$codepoints_len ->
    [one] { $identifier_type ->
        [Exclusion] a character from an archaic script
        [Technical] a character that is for non-linguistic, specialized usage
        [Limited_Use] a character from a script in limited use
        [Not_NFKC] a non normalized (NFKC) character
        *[other] an uncommon character
    }
    *[other] { $identifier_type ->
        [Exclusion] {$codepoints_len} characters from archaic scripts
        [Technical] {$codepoints_len} characters that are for non-linguistic, specialized usage
        [Limited_Use] {$codepoints_len} characters from scripts in limited use
        [Not_NFKC] {$codepoints_len} non normalized (NFKC) characters
        *[other] uncommon characters
    }
}: {$codepoints}
    .note = {$codepoints_len ->
        [one] this character is
        *[other] these characters are
    } included in the{$identifier_type ->
        [Restricted] {""}
        *[other] {" "}{$identifier_type}
    } Unicode general security profile

lint_if_let_dtor = {$dtor_kind ->
    [dyn] value may invoke a custom destructor because it contains a trait object
    *[concrete] value invokes this custom destructor
    }

lint_if_let_rescope = `if let` assigns a shorter lifetime since Edition 2024
    .label = this value has a significant drop implementation which may observe a major change in drop order and requires your discretion
    .help = the value is now dropped here in Edition 2024
    .suggestion = a `match` with a single arm can preserve the drop order up to Edition 2021

lint_ignored_unless_crate_specified = {$level}({$name}) is ignored unless specified at crate level

lint_ill_formed_attribute_input = {$num_suggestions ->
        [1] attribute must be of the form {$suggestions}
        *[other] valid forms for the attribute are {$suggestions}
    }

lint_impl_trait_overcaptures = `{$self_ty}` will capture more lifetimes than possibly intended in edition 2024
    .note = specifically, {$num_captured ->
        [one] this lifetime is
        *[other] these lifetimes are
     } in scope but not mentioned in the type's bounds
    .note2 = all lifetimes in scope will be captured by `impl Trait`s in edition 2024

lint_impl_trait_redundant_captures = all possible in-scope parameters are already captured, so `use<...>` syntax is redundant
    .suggestion = remove the `use<...>` syntax

lint_implicit_unsafe_autorefs = implicit autoref creates a reference to the dereference of a raw pointer
    .note = creating a reference requires the pointer target to be valid and imposes aliasing requirements
    .raw_ptr = this raw pointer has type `{$raw_ptr_ty}`
    .autoref = autoref is being applied to this expression, resulting in: `{$autoref_ty}`
    .overloaded_deref = references are created through calls to explicit `Deref(Mut)::deref(_mut)` implementations
    .method_def = method calls to `{$method_name}` require a reference
    .suggestion = try using a raw pointer method instead; or if this reference is intentional, make it explicit

lint_improper_ctypes = `extern` {$desc} uses type `{$ty}`, which is not FFI-safe
    .label = not FFI-safe
    .note = the type is defined here

lint_improper_ctypes_array_help = consider passing a pointer to the array

lint_improper_ctypes_array_reason = passing raw arrays by value is not FFI-safe
lint_improper_ctypes_box = box cannot be represented as a single pointer

lint_improper_ctypes_char_help = consider using `u32` or `libc::wchar_t` instead

lint_improper_ctypes_char_reason = the `char` type has no C equivalent

lint_improper_ctypes_cstr_help =
    consider passing a `*const std::ffi::c_char` instead, and use `CStr::as_ptr()`
lint_improper_ctypes_cstr_reason = `CStr`/`CString` do not have a guaranteed layout

lint_improper_ctypes_dyn = trait objects have no C equivalent

lint_improper_ctypes_enum_repr_help =
    consider adding a `#[repr(C)]`, `#[repr(transparent)]`, or integer `#[repr(...)]` attribute to this enum

lint_improper_ctypes_enum_repr_reason = enum has no representation hint
lint_improper_ctypes_fnptr_help = consider using an `extern fn(...) -> ...` function pointer instead

lint_improper_ctypes_fnptr_reason = this function pointer has Rust-specific calling convention
lint_improper_ctypes_non_exhaustive = this enum is non-exhaustive
lint_improper_ctypes_non_exhaustive_variant = this enum has non-exhaustive variants

lint_improper_ctypes_only_phantomdata = composed only of `PhantomData`

lint_improper_ctypes_opaque = opaque types have no C equivalent

lint_improper_ctypes_slice_help = consider using a raw pointer instead

lint_improper_ctypes_slice_reason = slices have no C equivalent
lint_improper_ctypes_str_help = consider using `*const u8` and a length instead

lint_improper_ctypes_str_reason = string slices have no C equivalent
lint_improper_ctypes_struct_fieldless_help = consider adding a member to this struct

lint_improper_ctypes_struct_fieldless_reason = this struct has no fields
lint_improper_ctypes_struct_layout_help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this struct

lint_improper_ctypes_struct_layout_reason = this struct has unspecified layout
lint_improper_ctypes_struct_non_exhaustive = this struct is non-exhaustive
lint_improper_ctypes_struct_zst = this struct contains only zero-sized fields

lint_improper_ctypes_tuple_help = consider using a struct instead

lint_improper_ctypes_tuple_reason = tuples have unspecified layout
lint_improper_ctypes_union_fieldless_help = consider adding a member to this union

lint_improper_ctypes_union_fieldless_reason = this union has no fields
lint_improper_ctypes_union_layout_help = consider adding a `#[repr(C)]` or `#[repr(transparent)]` attribute to this union

lint_improper_ctypes_union_layout_reason = this union has unspecified layout
lint_improper_ctypes_union_non_exhaustive = this union is non-exhaustive

lint_incomplete_include =
    include macro expected single expression in source

lint_inner_macro_attribute_unstable = inner macro attributes are unstable

lint_invalid_asm_label_binary = avoid using labels containing only the digits `0` and `1` in inline assembly
    .label = use a different label that doesn't start with `0` or `1`
    .help = start numbering with `2` instead
    .note1 = an LLVM bug makes these labels ambiguous with a binary literal number on x86
    .note2 = see <https://github.com/llvm/llvm-project/issues/99547> for more information

lint_invalid_asm_label_format_arg = avoid using named labels in inline assembly
    .help = only local labels of the form `<number>:` should be used in inline asm
    .note1 = format arguments may expand to a non-numeric value
    .note2 = see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information
lint_invalid_asm_label_named = avoid using named labels in inline assembly
    .help = only local labels of the form `<number>:` should be used in inline asm
    .note = see the asm section of Rust By Example <https://doc.rust-lang.org/nightly/rust-by-example/unsafe/asm.html#labels> for more information
lint_invalid_asm_label_no_span = the label may be declared in the expansion of a macro
lint_invalid_crate_type_value = invalid `crate_type` value
    .suggestion = did you mean

# FIXME: we should ordinalize $valid_up_to when we add support for doing so
lint_invalid_from_utf8_checked = calls to `{$method}` with an invalid literal always return an error
    .label = the literal was valid UTF-8 up to the {$valid_up_to} bytes

# FIXME: we should ordinalize $valid_up_to when we add support for doing so
lint_invalid_from_utf8_unchecked = calls to `{$method}` with an invalid literal are undefined behavior
    .label = the literal was valid UTF-8 up to the {$valid_up_to} bytes

lint_invalid_nan_comparisons_eq_ne = incorrect NaN comparison, NaN cannot be directly compared to itself
    .suggestion = use `f32::is_nan()` or `f64::is_nan()` instead

lint_invalid_nan_comparisons_lt_le_gt_ge = incorrect NaN comparison, NaN is not orderable

lint_invalid_null_arguments = calling this function with a null pointer is undefined behavior, even if the result of the function is unused
    .origin = null pointer originates from here
    .doc = for more information, visit <https://doc.rust-lang.org/std/ptr/index.html> and <https://doc.rust-lang.org/reference/behavior-considered-undefined.html>

lint_invalid_reference_casting_assign_to_ref = assigning to `&T` is undefined behavior, consider using an `UnsafeCell`
    .label = casting happened here

lint_invalid_reference_casting_bigger_layout = casting references to a bigger memory layout than the backing allocation is undefined behavior, even if the reference is unused
    .label = casting happened here
    .alloc = backing allocation comes from here
    .layout = casting from `{$from_ty}` ({$from_size} bytes) to `{$to_ty}` ({$to_size} bytes)

lint_invalid_reference_casting_borrow_as_mut = casting `&T` to `&mut T` is undefined behavior, even if the reference is unused, consider instead using an `UnsafeCell`
    .label = casting happened here

lint_invalid_reference_casting_note_book = for more information, visit <https://doc.rust-lang.org/book/ch15-05-interior-mutability.html>

lint_invalid_reference_casting_note_ty_has_interior_mutability = even for types with interior mutability, the only legal way to obtain a mutable pointer from a shared reference is through `UnsafeCell::get`

lint_legacy_derive_helpers = derive helper attribute is used before it is introduced
    .label = the attribute is introduced here

lint_lintpass_by_hand = implementing `LintPass` by hand
    .help = try using `declare_lint_pass!` or `impl_lint_pass!` instead

lint_macro_expanded_macro_exports_accessed_by_absolute_paths = macro-expanded `macro_export` macros from the current crate cannot be referred to by absolute paths
    .note = the macro is defined here

lint_macro_expr_fragment_specifier_2024_migration =
    the `expr` fragment specifier will accept more expressions in the 2024 edition
    .suggestion = to keep the existing behavior, use the `expr_2021` fragment specifier
lint_macro_is_private = macro `{$ident}` is private

lint_macro_rule_never_used = rule #{$n} of macro `{$name}` is never used

lint_macro_use_deprecated =
    applying the `#[macro_use]` attribute to an `extern crate` item is deprecated
    .help = remove it and import macros at use sites with a `use` item instead

lint_malformed_attribute = malformed lint attribute input

lint_map_unit_fn = `Iterator::map` call that discard the iterator's values
    .note = `Iterator::map`, like many of the methods on `Iterator`, gets executed lazily, meaning that its effects won't be visible until it is iterated
    .function_label = this function returns `()`, which is likely not what you wanted
    .argument_label = called `Iterator::map` with callable that returns `()`
    .map_label = after this call to map, the resulting iterator is `impl Iterator<Item = ()>`, which means the only information carried by the iterator is the number of items
    .suggestion = you might have meant to use `Iterator::for_each`

lint_metavariable_still_repeating = variable `{$name}` is still repeating at this depth

lint_metavariable_wrong_operator = meta-variable repeats with different Kleene operator

lint_mismatched_lifetime_syntaxes =
    lifetime flowing from input to output with different syntax can be confusing
    .label_mismatched_lifetime_syntaxes_inputs =
        {$n_inputs ->
            [one] this lifetime flows
            *[other] these lifetimes flow
        } to the output
    .label_mismatched_lifetime_syntaxes_outputs =
        the {$n_outputs ->
            [one] lifetime gets
            *[other] lifetimes get
        } resolved as `{$lifetime_name}`

lint_mismatched_lifetime_syntaxes_suggestion_explicit =
    one option is to consistently use `{$lifetime_name}`

lint_mismatched_lifetime_syntaxes_suggestion_implicit =
    one option is to consistently remove the lifetime

lint_mismatched_lifetime_syntaxes_suggestion_mixed =
    one option is to remove the lifetime for references and use the anonymous lifetime for paths

lint_missing_fragment_specifier = missing fragment specifier

lint_missing_unsafe_on_extern = extern blocks should be unsafe
    .suggestion = needs `unsafe` before the extern keyword

lint_mixed_script_confusables =
    the usage of Script Group `{$set}` in this crate consists solely of mixed script confusables
    .includes_note = the usage includes {$includes}
    .note = please recheck to make sure their usages are indeed what you want

lint_multiple_supertrait_upcastable = `{$ident}` is dyn-compatible and has multiple supertraits

lint_named_argument_used_positionally = named argument `{$named_arg_name}` is not used by name
    .label_named_arg = this named argument is referred to by position in formatting string
    .label_position_arg = this formatting argument uses named argument `{$named_arg_name}` by position
    .suggestion = use the named argument by name to avoid ambiguity

lint_node_source = `forbid` level set here
    .note = {$reason}

lint_non_binding_let_multi_drop_fn =
    consider immediately dropping the value using `drop(..)` after the `let` statement

lint_non_binding_let_multi_suggestion =
    consider immediately dropping the value

lint_non_binding_let_on_drop_type =
    non-binding let on a type that has a destructor

lint_non_binding_let_on_sync_lock = non-binding let on a synchronization lock
    .label = this lock is not assigned to a binding and is immediately dropped

lint_non_binding_let_suggestion =
    consider binding to an unused variable to avoid immediately dropping the value

lint_non_camel_case_type = {$sort} `{$name}` should have an upper camel case name
    .suggestion = convert the identifier to upper camel case
    .label = should have an UpperCamelCase name

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

lint_non_fmt_panic_braces =
    panic message contains {$count ->
        [one] a brace
        *[other] braces
    }
    .note = this message is not used as a format string, but will be in Rust 2021
    .suggestion = add a "{"{"}{"}"}" format string to use the message literally

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

lint_non_glob_import_type_ir_inherent = non-glob import of `rustc_type_ir::inherent`
    .suggestion = try using a glob import instead

lint_non_local_definitions_cargo_update = the {$macro_kind} `{$macro_name}` may come from an old version of the `{$crate_name}` crate, try updating your dependency with `cargo update -p {$crate_name}`

lint_non_local_definitions_impl = non-local `impl` definition, `impl` blocks should be written at the same level as their item
    .non_local = an `impl` is never scoped, even when it is nested inside an item, as it may impact type checking outside of that item, which can be the case if neither the trait or the self type are at the same nesting level as the `impl`
    .doctest = make this doc-test a standalone test with its own `fn main() {"{"} ... {"}"}`
    .exception = items in an anonymous const item (`const _: () = {"{"} ... {"}"}`) are treated as in the same scope as the anonymous const's declaration for the purpose of this lint
    .const_anon = use a const-anon item to suppress this lint
    .macro_to_change = the {$macro_kind} `{$macro_to_change}` defines the non-local `impl`, and may need to be changed

lint_non_local_definitions_impl_move_help =
    move the `impl` block outside of this {$body_kind_descr} {$depth ->
        [one] `{$body_name}`
       *[other] `{$body_name}` and up {$depth} bodies
    }

lint_non_local_definitions_macro_rules = non-local `macro_rules!` definition, `#[macro_export]` macro should be written at top level module
    .help =
        remove the `#[macro_export]` or move this `macro_rules!` outside the of the current {$body_kind_descr} {$depth ->
            [one] `{$body_name}`
           *[other] `{$body_name}` and up {$depth} bodies
        }
    .help_doctest =
        remove the `#[macro_export]` or make this doc-test a standalone test with its own `fn main() {"{"} ... {"}"}`
    .non_local = a `macro_rules!` definition is non-local if it is nested inside an item and has a `#[macro_export]` attribute

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
    .suggestion = remove this redundant call
    .note = the type `{$orig_ty}` does not implement `{$trait_}`, so calling `{$method}` on `&{$orig_ty}` copies the reference, which does not do anything and can be removed
    .derive_suggestion = if you meant to clone `{$orig_ty}`, implement `Clone` for it

lint_only_cast_u8_to_char = only `u8` can be cast into `char`
    .suggestion = use a `char` literal instead

lint_opaque_hidden_inferred_bound = opaque type `{$ty}` does not satisfy its associated type bounds
    .specifically = this associated type bound is unsatisfied for `{$proj_ty}`

lint_opaque_hidden_inferred_bound_sugg = add this bound

lint_or_patterns_back_compat = the meaning of the `pat` fragment specifier is changing in Rust 2021, which may affect this macro
    .suggestion = use pat_param to preserve semantics

lint_out_of_scope_macro_calls = cannot find macro `{$path}` in the current scope when looking from {$location}
    .label = not found from {$location}
    .help = import `macro_rules` with `use` to make it callable above its definition

lint_overflowing_bin_hex = literal out of range for `{$ty}`
    .negative_note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}`
    .negative_becomes_note = and the value `-{$lit}` will become `{$actually}{$ty}`
    .positive_note = the literal `{$lit}` (decimal `{$dec}`) does not fit into the type `{$ty}` and will become `{$actually}{$ty}`
    .suggestion = consider using the type `{$suggestion_ty}` instead
    .sign_bit_suggestion = to use as a negative number (decimal `{$negative_val}`), consider using the type `{$uint_ty}` for the literal and cast it to `{$int_ty}`
    .help = consider using the type `{$suggestion_ty}` instead

lint_overflowing_int = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`
    .help = consider using the type `{$suggestion_ty}` instead

lint_overflowing_literal = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` and will be converted to `{$ty}::INFINITY`

lint_overflowing_uint = literal out of range for `{$ty}`
    .note = the literal `{$lit}` does not fit into the type `{$ty}` whose range is `{$min}..={$max}`

lint_overruled_attribute = {$lint_level}({$lint_source}) incompatible with previous forbid
    .label = overruled by previous forbid

lint_pass_by_value = passing `{$ty}` by reference
    .suggestion = try passing by value

lint_path_statement_drop = path statement drops value
    .suggestion = use `drop` to clarify the intent

lint_path_statement_no_effect = path statement with no effect

lint_pattern_in_bodiless = patterns aren't allowed in functions without bodies
    .label = pattern not allowed in function without body

lint_pattern_in_foreign = patterns aren't allowed in foreign function declarations
    .label = pattern not allowed in foreign function

lint_private_extern_crate_reexport = extern crate `{$ident}` is private and cannot be re-exported
    .suggestion = consider making the `extern crate` item publicly accessible

lint_proc_macro_derive_resolution_fallback = cannot find {$ns} `{$ident}` in this scope
    .label = names from parent modules are not accessible without an explicit import

lint_query_instability = using `{$query}` can result in unstable query results
    .note = if you believe this case to be fine, allow this lint and add a comment explaining your rationale

lint_query_untracked = `{$method}` accesses information that is not tracked by the query system
    .note = if you believe this case to be fine, allow this lint and add a comment explaining your rationale

lint_range_endpoint_out_of_range = range endpoint is out of range for `{$ty}`

lint_range_use_inclusive_range = use an inclusive range instead


lint_raw_prefix = prefix `'r` is reserved
    .label = reserved prefix
    .suggestion = insert whitespace here to avoid this being parsed as a prefix in Rust 2021

lint_reason_must_be_string_literal = reason must be a string literal

lint_reason_must_come_last = reason in lint attribute must come last

lint_redundant_import = the item `{$ident}` is imported redundantly
    .label_imported_here = the item `{$ident}` is already imported here
    .label_defined_here = the item `{$ident}` is already defined here
    .label_imported_prelude = the item `{$ident}` is already imported by the extern prelude
    .label_defined_prelude = the item `{$ident}` is already defined by the extern prelude

lint_redundant_import_visibility = glob import doesn't reexport anything with visibility `{$import_vis}` because no imported item is public enough
    .note = the most public imported item is `{$max_vis}`
    .help = reduce the glob import's visibility or increase visibility of imported items

lint_redundant_semicolons =
    unnecessary trailing {$multiple ->
        [true] semicolons
        *[false] semicolon
    }
    .suggestion = remove {$multiple ->
        [true] these semicolons
        *[false] this semicolon
    }

lint_remove_mut_from_pattern = remove `mut` from the parameter

lint_removed_lint = lint `{$name}` has been removed: {$reason}

lint_renamed_lint = lint `{$name}` has been renamed to `{$replace}`
    .suggestion = use the new name
    .help = use the new name `{$replace}`

lint_requested_level = requested on the command line with `{$level} {$lint_name}`

lint_reserved_multihash = reserved token in Rust 2024
    .suggestion = insert whitespace here to avoid this being parsed as a forbidden token in Rust 2024

lint_reserved_prefix = prefix `{$prefix}` is unknown
    .label = unknown prefix
    .suggestion = insert whitespace here to avoid this being parsed as a prefix in Rust 2021

lint_reserved_string = will be parsed as a guarded string in Rust 2024
    .suggestion = insert whitespace here to avoid this being parsed as a guarded string in Rust 2024

lint_shadowed_into_iter =
    this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to `<{$target} as IntoIterator>::into_iter` in Rust {$edition}
    .use_iter_suggestion = use `.iter()` instead of `.into_iter()` to avoid ambiguity
    .remove_into_iter_suggestion = or remove `.into_iter()` to iterate by value
    .use_explicit_into_iter_suggestion =
        or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value

lint_single_use_lifetime = lifetime parameter `{$ident}` only used once
    .label_param = this lifetime...
    .label_use = ...is used only here
    .suggestion = elide the single-use lifetime

lint_span_use_eq_ctxt = use `.eq_ctxt()` instead of `.ctxt() == .ctxt()`

lint_static_mut_refs_lint = creating a {$shared_label}reference to mutable static
    .label = {$shared_label}reference to mutable static
    .suggestion = use `&raw const` instead to create a raw pointer
    .suggestion_mut = use `&raw mut` instead to create a raw pointer
    .shared_note = shared references to mutable statics are dangerous; it's undefined behavior if the static is mutated or if a mutable reference is created for it while the shared reference lives
    .mut_note = mutable references to mutable statics are dangerous; it's undefined behavior if any other pointer to the static is used or if any other reference is created for the static while the mutable reference lives

lint_supertrait_as_deref_target = this `Deref` implementation is covered by an implicit supertrait coercion
    .label = `{$self_ty}` implements `Deref<Target = dyn {$target_principal}>` which conflicts with supertrait `{$supertrait_principal}`
    .label2 = target type is a supertrait of `{$self_ty}`
    .help = consider removing this implementation or replacing it with a method instead

lint_suspicious_double_ref_clone =
    using `.clone()` on a double reference, which returns `{$ty}` instead of cloning the inner type

lint_suspicious_double_ref_deref =
    using `.deref()` on a double reference, which returns `{$ty}` instead of dereferencing the inner type

lint_symbol_intern_string_literal = using `Symbol::intern` on a string literal
    .help = consider adding the symbol to `compiler/rustc_span/src/symbol.rs`

lint_trailing_semi_macro = trailing semicolon in macro used in expression position
    .note1 = macro invocations at the end of a block are treated as expressions
    .note2 = to ignore the value produced by the macro, add a semicolon after the invocation of `{$name}`

lint_ty_qualified = usage of qualified `ty::{$ty}`
    .suggestion = try importing it and using it unqualified

lint_tykind = usage of `ty::TyKind`
    .help = try using `Ty` instead

lint_tykind_kind = usage of `ty::TyKind::<kind>`
    .suggestion = try using `ty::<kind>` directly

lint_type_ir_inherent_usage = do not use `rustc_type_ir::inherent` unless you're inside of the trait solver
    .note = the method or struct you're looking for is likely defined somewhere else downstream in the compiler

lint_type_ir_trait_usage = do not use `rustc_type_ir::Interner` or `rustc_type_ir::InferCtxtLike` unless you're inside of the trait solver
    .note = the method or struct you're looking for is likely defined somewhere else downstream in the compiler

lint_undefined_transmute = pointers cannot be transmuted to integers during const eval
    .note = at compile-time, pointers do not have an integer value
    .note2 = avoiding this restriction via `union` or raw pointers leads to compile-time undefined behavior
    .help = for more information, see https://doc.rust-lang.org/std/mem/fn.transmute.html

lint_undropped_manually_drops = calls to `std::mem::drop` with `std::mem::ManuallyDrop` instead of the inner value does nothing
    .label = argument has type `{$arg_ty}`
    .suggestion = use `std::mem::ManuallyDrop::into_inner` to get the inner value

lint_unexpected_builtin_cfg = unexpected `--cfg {$cfg}` flag
    .controlled_by = config `{$cfg_name}` is only supposed to be controlled by `{$controlled_by}`
    .incoherent = manually setting a built-in cfg can and does create incoherent behaviors

lint_unexpected_cfg_add_build_rs_println = or consider adding `{$build_rs_println}` to the top of the `build.rs`
lint_unexpected_cfg_add_cargo_feature = consider using a Cargo feature instead
lint_unexpected_cfg_add_cargo_toml_lint_cfg = or consider adding in `Cargo.toml` the `check-cfg` lint config for the lint:{$cargo_toml_lint_cfg}
lint_unexpected_cfg_add_cmdline_arg = to expect this configuration use `{$cmdline_arg}`
lint_unexpected_cfg_cargo_update = the {$macro_kind} `{$macro_name}` may come from an old version of the `{$crate_name}` crate, try updating your dependency with `cargo update -p {$crate_name}`

lint_unexpected_cfg_define_features = consider defining some features in `Cargo.toml`
lint_unexpected_cfg_doc_cargo = see <https://doc.rust-lang.org/nightly/rustc/check-cfg/cargo-specifics.html> for more information about checking conditional configuration
lint_unexpected_cfg_doc_rustc = see <https://doc.rust-lang.org/nightly/rustc/check-cfg.html> for more information about checking conditional configuration

lint_unexpected_cfg_from_external_macro_origin = using a cfg inside a {$macro_kind} will use the cfgs from the destination crate and not the ones from the defining crate
lint_unexpected_cfg_from_external_macro_refer = try referring to `{$macro_name}` crate for guidance on how handle this unexpected cfg
lint_unexpected_cfg_name = unexpected `cfg` condition name: `{$name}`
lint_unexpected_cfg_name_expected_names = expected names are: {$possibilities}{$and_more ->
        [0] {""}
        *[other] {" "}and {$and_more} more
    }
lint_unexpected_cfg_name_expected_values = expected values for `{$best_match}` are: {$possibilities}
lint_unexpected_cfg_name_similar_name = there is a config with a similar name
lint_unexpected_cfg_name_similar_name_different_values = there is a config with a similar name and different values
lint_unexpected_cfg_name_similar_name_no_value = there is a config with a similar name and no value
lint_unexpected_cfg_name_similar_name_value = there is a config with a similar name and value
lint_unexpected_cfg_name_version_syntax = there is a similar config predicate: `version("..")`
lint_unexpected_cfg_name_with_similar_value = found config with similar value

lint_unexpected_cfg_value = unexpected `cfg` condition value: {$has_value ->
        [true] `{$value}`
        *[false] (none)
    }
lint_unexpected_cfg_value_add_feature = consider adding `{$value}` as a feature in `Cargo.toml`
lint_unexpected_cfg_value_expected_values = expected values for `{$name}` are: {$have_none_possibility ->
        [true] {"(none), "}
        *[false] {""}
    }{$possibilities}{$and_more ->
        [0] {""}
        *[other] {" "}and {$and_more} more
    }
lint_unexpected_cfg_value_no_expected_value = no expected value for `{$name}`
lint_unexpected_cfg_value_no_expected_values = no expected values for `{$name}`
lint_unexpected_cfg_value_remove_condition = remove the condition
lint_unexpected_cfg_value_remove_value = remove the value
lint_unexpected_cfg_value_similar_name = there is a expected value with a similar name
lint_unexpected_cfg_value_specify_value = specify a config value

lint_ungated_async_fn_track_caller = `#[track_caller]` on async functions is a no-op
     .label = this function will not propagate the caller location

lint_unicode_text_flow = unicode codepoint changing visible direction of text present in comment
    .label = {$num_codepoints ->
            [1] this comment contains an invisible unicode text flow control codepoint
            *[other] this comment contains invisible unicode text flow control codepoints
        }
    .note = these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen
    .suggestion = if their presence wasn't intentional, you can remove them
    .label_comment_char = {$c_debug}


lint_unit_bindings = binding has unit type `()`
    .label = this pattern is inferred to be the unit type `()`

lint_unknown_diagnostic_attribute = unknown diagnostic attribute
lint_unknown_diagnostic_attribute_typo_sugg = an attribute with a similar name exists

lint_unknown_gated_lint =
    unknown lint: `{$name}`
    .note = the `{$name}` lint is unstable

lint_unknown_lint =
    unknown lint: `{$name}`
    .suggestion = {$from_rustc ->
        [true] a lint with a similar name exists in `rustc` lints
        *[false] did you mean
    }
    .help = {$from_rustc ->
        [true] a lint with a similar name exists in `rustc` lints: `{$replace}`
        *[false] did you mean: `{$replace}`
    }

lint_unknown_macro_variable = unknown macro variable `{$name}`

lint_unknown_tool_in_scoped_lint = unknown tool name `{$tool_name}` found in scoped lint: `{$tool_name}::{$lint_name}`
    .help = add `#![register_tool({$tool_name})]` to the crate root

lint_unnameable_test_items = cannot test inner items

lint_unnecessary_qualification = unnecessary qualification
    .suggestion = remove the unnecessary path segments

lint_unpredictable_fn_pointer_comparisons = function pointer comparisons do not produce meaningful results since their addresses are not guaranteed to be unique
    .note_duplicated_fn = the address of the same function can vary between different codegen units
    .note_deduplicated_fn = furthermore, different functions could have the same address after being merged together
    .note_visit_fn_addr_eq = for more information visit <https://doc.rust-lang.org/nightly/core/ptr/fn.fn_addr_eq.html>
    .fn_addr_eq_suggestion = refactor your code, or use `std::ptr::fn_addr_eq` to suppress the lint

lint_unqualified_local_imports = `use` of a local item without leading `self::`, `super::`, or `crate::`

lint_unsafe_attr_outside_unsafe = unsafe attribute used without unsafe
    .label = usage of unsafe attribute
lint_unsafe_attr_outside_unsafe_suggestion = wrap the attribute in `unsafe(...)`

lint_unsupported_group = `{$lint_group}` lint group is not supported with ´--force-warn´

lint_untranslatable_diag = diagnostics should be created using translatable messages

lint_unused_allocation = unnecessary allocation, use `&` instead
lint_unused_allocation_mut = unnecessary allocation, use `&mut` instead

lint_unused_builtin_attribute = unused attribute `{$attr_name}`
    .note = the built-in attribute `{$attr_name}` will be ignored, since it's applied to the macro invocation `{$macro_name}`

lint_unused_closure =
    unused {$pre}{$count ->
        [one] closure
        *[other] closures
    }{$post} that must be used
    .note = closures are lazy and do nothing unless called

lint_unused_comparisons = comparison is useless due to type limits

lint_unused_coroutine =
    unused {$pre}{$count ->
        [one] coroutine
        *[other] coroutine
    }{$post} that must be used
    .note = coroutines are lazy and do nothing unless resumed

lint_unused_crate_dependency = extern crate `{$extern_crate}` is unused in crate `{$local_crate}`
    .help = remove the dependency or add `use {$extern_crate} as _;` to the crate root

lint_unused_def = unused {$pre}`{$def}`{$post} that must be used
    .suggestion = use `let _ = ...` to ignore the resulting value

lint_unused_delim = unnecessary {$delim} around {$item}
    .suggestion = remove these {$delim}

lint_unused_doc_comment = unused doc comment
    .label = rustdoc does not generate documentation for macro invocations
    .help = to document an item produced by a macro, the macro must produce the documentation as part of its expansion

lint_unused_extern_crate = unused extern crate
    .label = unused
    .suggestion = remove the unused `extern crate`

lint_unused_import_braces = braces around {$node} is unnecessary

lint_unused_imports = {$num_snippets ->
        [one] unused import: {$span_snippets}
        *[other] unused imports: {$span_snippets}
    }
    .suggestion_remove_whole_use = remove the whole `use` item
    .suggestion_remove_imports = {$num_to_remove ->
            [one] remove the unused import
            *[other] remove the unused imports
        }
    .help = if this is a test module, consider adding a `#[cfg(test)]` to the containing module

lint_unused_label = unused label

lint_unused_lifetime = lifetime parameter `{$ident}` never used
    .suggestion = elide the unused lifetime

lint_unused_macro_definition = unused macro definition: `{$name}`

lint_unused_macro_use = unused `#[macro_use]` import

lint_unused_op = unused {$op} that must be used
    .label = the {$op} produces a value
    .suggestion = use `let _ = ...` to ignore the resulting value

lint_unused_result = unused result of type `{$ty}`

lint_use_let_underscore_ignore_suggestion = use `let _ = ...` to ignore the expression or result

lint_useless_ptr_null_checks_fn_ptr = function pointers are not nullable, so checking them for null will always return false
    .help = wrap the function pointer inside an `Option` and use `Option::is_none` to check for null pointer value
    .label = expression has type `{$orig_ty}`

lint_useless_ptr_null_checks_fn_ret = returned pointer of `{$fn_name}` call is never null, so checking it for null will always return false

lint_useless_ptr_null_checks_ref = references are not nullable, so checking them for null will always return false
    .label = expression has type `{$orig_ty}`

lint_uses_power_alignment = repr(C) does not follow the power alignment rule. This may affect platform C ABI compatibility for this type

lint_variant_size_differences =
    enum variant is more than three times larger ({$largest} bytes) than the next largest
