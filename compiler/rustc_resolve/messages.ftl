resolve_accessible_unsure = not sure whether the path is accessible or not
    .note = the type may have associated items, but we are currently not checking them

resolve_add_as_non_derive =
    add as non-Derive macro
    `#[{$macro_path}]`

resolve_added_macro_use =
    have you added the `#[macro_use]` on the module/import?

resolve_ancestor_only =
    visibilities can only be restricted to ancestor modules

resolve_anonymous_lifetime_non_gat_report_error =
    in the trait associated type is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type
    .label = this lifetime must come from the implemented type

resolve_arguments_macro_use_not_allowed = arguments to `macro_use` are not allowed here

resolve_associated_const_with_similar_name_exists =
    there is an associated constant with a similar name

resolve_associated_fn_with_similar_name_exists =
    there is an associated function with a similar name

resolve_associated_type_with_similar_name_exists =
    there is an associated type with a similar name

resolve_attempt_to_use_non_constant_value_in_constant =
    attempt to use a non-constant value in a constant

resolve_attempt_to_use_non_constant_value_in_constant_label_with_suggestion =
    non-constant value

resolve_attempt_to_use_non_constant_value_in_constant_with_suggestion =
    consider using `{$suggestion}` instead of `{$current}`

resolve_attempt_to_use_non_constant_value_in_constant_without_suggestion =
    this would need to be a `{$suggestion}`

resolve_attributes_starting_with_rustc_are_reserved =
    attributes starting with `rustc` are reserved for use by the `rustc` compiler

resolve_bad_macro_import = bad macro import

resolve_binding_in_never_pattern =
    never patterns cannot contain variable bindings
    .suggestion = use a wildcard `_` instead

resolve_binding_shadows_something_unacceptable =
    {$shadowing_binding}s cannot shadow {$shadowed_binding}s
    .label = cannot be named the same as {$article} {$shadowed_binding}
    .label_shadowed_binding = the {$shadowed_binding} `{$name}` is {$participle} here

resolve_binding_shadows_something_unacceptable_suggestion =
    try specify the pattern arguments

resolve_cannot_be_reexported_crate_public =
    `{$ident}` is only public within the crate, and cannot be re-exported outside

resolve_cannot_be_reexported_private =
    `{$ident}` is private, and cannot be re-exported

resolve_cannot_capture_dynamic_environment_in_fn_item =
    can't capture dynamic environment in a fn item
    .help = use the `|| {"{"} ... {"}"}` closure form instead

resolve_cannot_determine_import_resolution =
    cannot determine resolution for the import
    .note = import resolution is stuck, try simplifying other imports

resolve_cannot_determine_macro_resolution =
    cannot determine resolution for the {$kind} `{$path}`
    .note = import resolution is stuck, try simplifying macro imports

resolve_cannot_find_builtin_macro_with_name =
    cannot find a built-in macro with name `{$ident}`

resolve_cannot_find_ident_in_this_scope =
    cannot find {$expected} `{$ident}` in this scope

resolve_cannot_glob_import_possible_crates =
    cannot glob-import all possible crates

resolve_cannot_use_through_an_import =
    cannot use {$article} {$descr} through an import
    .note = the {$descr} imported here

resolve_change_import_binding =
    you can use `as` to change the binding name of the import

resolve_consider_adding_a_derive =
    consider adding a derive

resolve_consider_adding_macro_export =
    consider adding a `#[macro_export]` to the macro in the imported module

resolve_consider_declaring_with_pub =
    consider declaring type or module `{$ident}` with `pub`

resolve_consider_making_the_field_public =
    { $number_of_fields ->
        [one] consider making the field publicly accessible
        *[other] consider making the fields publicly accessible
    }

resolve_consider_marking_as_pub =
    consider marking `{$ident}` as `pub` in the imported module

resolve_consider_move_macro_position =
    consider moving the definition of `{$ident}` before this call


resolve_const_not_member_of_trait =
    const `{$const_}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_const_param_in_enum_discriminant =
    const parameters may not be used in enum discriminant values

resolve_const_param_in_non_trivial_anon_const =
    const parameters may only be used as standalone arguments here, i.e. `{$name}`

resolve_constructor_private_if_any_field_private =
    a constructor is private if any of the fields is private

resolve_elided_anonymous_lifetime_report_error =
    `&` without an explicit lifetime name cannot be used here
    .label = explicit lifetime name needed here

resolve_elided_anonymous_lifetime_report_error_suggestion =
    consider introducing a higher-ranked lifetime here

resolve_expected_module_found =
    expected module, found {$res} `{$path_str}`
    .label = not a module

resolve_explicit_anonymous_lifetime_report_error =
    `'_` cannot be used here
    .label = `'_` is a reserved lifetime name

resolve_explicit_unsafe_traits =
    unsafe traits like `{$ident}` should be implemented explicitly

resolve_extern_crate_loading_macro_not_at_crate_root =
    an `extern crate` loading macros must be at the crate root

resolve_extern_crate_self_requires_renaming =
    `extern crate self;` requires renaming
    .suggestion = rename the `self` crate to be able to import it

resolve_forward_declared_generic_in_const_param_ty =
    const parameter types cannot reference parameters before they are declared
    .label = const parameter type cannot reference `{$param}` before it is declared

resolve_forward_declared_generic_param =
    generic parameter defaults cannot reference parameters before they are declared
    .label = cannot reference `{$param}` before it is declared

resolve_found_an_item_configured_out =
    found an item that was configured out

resolve_generic_arguments_in_macro_path =
    generic arguments in macro path

resolve_generic_params_from_outer_item =
    can't use {$is_self ->
        [true] `Self`
        *[false] generic parameters
    } from outer item
    .label = use of {$is_self ->
        [true] `Self`
        *[false] generic parameter
    } from outer item
    .refer_to_type_directly = refer to the type directly here instead
    .suggestion = try introducing a local generic parameter here

resolve_generic_params_from_outer_item_const = a `const` is a separate item from the item that contains it

resolve_generic_params_from_outer_item_const_param = const parameter from outer item

resolve_generic_params_from_outer_item_self_ty_alias = `Self` type implicitly declared here, by this `impl`

resolve_generic_params_from_outer_item_self_ty_param = can't use `Self` here

resolve_generic_params_from_outer_item_static = a `static` is a separate item from the item that contains it

resolve_generic_params_from_outer_item_ty_param = type parameter from outer item

resolve_ident_bound_more_than_once_in_parameter_list =
    identifier `{$identifier}` is bound more than once in this parameter list
    .label = used as parameter more than once

resolve_ident_bound_more_than_once_in_same_pattern =
    identifier `{$identifier}` is bound more than once in the same pattern
    .label = used in a pattern more than once

resolve_ident_imported_here_but_it_is_desc =
    `{$imported_ident}` is imported here, but it is {$imported_ident_desc}

resolve_ident_in_scope_but_it_is_desc =
    `{$imported_ident}` is in scope, but it is {$imported_ident_desc}

resolve_implicit_elided_lifetimes_not_allowed_here = implicit elided lifetime not allowed here

resolve_imported_crate = `$crate` may not be imported

resolve_imported_macro_not_found = imported macro not found

resolve_imports_cannot_refer_to =
    imports cannot refer to {$what}

resolve_indeterminate =
    cannot determine resolution for the visibility

resolve_invalid_asm_sym =
    invalid `sym` operand
    .label = is a local variable
    .help = `sym` operands must refer to either a function or a static

resolve_is_private =
    {$ident_descr} `{$ident}` is private
    .label = private {$ident_descr}

resolve_item_was_behind_feature =
    the item is gated behind the `{$feature}` feature

resolve_item_was_cfg_out = the item is gated here

resolve_label_with_similar_name_reachable =
    a label with a similar name is reachable

resolve_lending_iterator_report_error =
    associated type `Iterator::Item` is declared without lifetime parameters, so using a borrowed type for them requires that lifetime to come from the implemented type
    .note = you can't create an `Iterator` that borrows each `Item` from itself, but you can instead create a new type that borrows your existing type and implement `Iterator` for that new type

resolve_lifetime_param_in_enum_discriminant =
    lifetime parameters may not be used in enum discriminant values

resolve_lifetime_param_in_non_trivial_anon_const =
    lifetime parameters may not be used in const expressions

resolve_lowercase_self =
    attempt to use a non-constant value in a constant
    .suggestion = try using `Self`

resolve_macro_cannot_use_as_attr =
    `{$ident}` exists, but a declarative macro cannot be used as an attribute macro

resolve_macro_cannot_use_as_derive =
     `{$ident}` exists, but a declarative macro cannot be used as a derive macro

resolve_macro_defined_later =
    a macro with the same name exists, but it appears later

resolve_macro_expanded_extern_crate_cannot_shadow_extern_arguments =
    macro-expanded `extern crate` items cannot shadow names passed with `--extern`

resolve_macro_expected_found =
    expected {$expected}, found {$found} `{$macro_path}`
    .label = not {$article} {$expected}

resolve_macro_extern_deprecated =
    `#[macro_escape]` is a deprecated synonym for `#[macro_use]`
    .help = try an outer attribute: `#[macro_use]`

resolve_macro_use_extern_crate_self = `#[macro_use]` is not supported on `extern crate self`

resolve_macro_use_name_already_in_use =
    `{$name}` is already in scope
    .note = macro-expanded `#[macro_use]`s may not shadow existing macros (see RFC 1560)

resolve_method_not_member_of_trait =
    method `{$method}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_missing_macro_rules_name = maybe you have forgotten to define a name for this `macro_rules!`

resolve_module_only =
    visibility must resolve to a module

resolve_name_defined_multiple_time =
    the name `{$name}` is defined multiple times
    .note = `{$name}` must be defined only once in the {$descr} namespace of this {$container}

resolve_name_defined_multiple_time_old_binding_definition =
    previous definition of the {$old_kind} `{$name}` here

resolve_name_defined_multiple_time_old_binding_import =
    previous import of the {$old_kind} `{$name}` here

resolve_name_defined_multiple_time_redefined =
    `{$name}` redefined here

resolve_name_defined_multiple_time_reimported =
    `{$name}` reimported here

resolve_name_is_already_used_as_generic_parameter =
    the name `{$name}` is already used for a generic parameter in this item's generic parameters
    .label = already used
    .first_use_of_name = first use of `{$name}`

resolve_name_reserved_in_attribute_namespace =
    name `{$ident}` is reserved in attribute namespace

resolve_note_and_refers_to_the_item_defined_here =
    {$first ->
        [true] {$dots ->
            [true] the {$binding_descr} `{$binding_name}` is defined here...
            *[false] the {$binding_descr} `{$binding_name}` is defined here
        }
        *[false] {$dots ->
            [true] ...and refers to the {$binding_descr} `{$binding_name}` which is defined here...
            *[false] ...and refers to the {$binding_descr} `{$binding_name}` which is defined here
        }
    }

resolve_outer_ident_is_not_publicly_reexported =
    {$outer_ident_descr} `{$outer_ident}` is not publicly re-exported

resolve_param_in_enum_discriminant =
    generic parameters may not be used in enum discriminant values
    .label = cannot perform const operation using `{$name}`

resolve_param_in_non_trivial_anon_const =
    generic parameters may not be used in const operations
    .label = cannot perform const operation using `{$name}`

resolve_param_in_non_trivial_anon_const_help =
    add `#![feature(generic_const_exprs)]` to allow generic const expressions

resolve_param_in_ty_of_const_param =
    the type of const parameters must not depend on other generic parameters
    .label = the type must not depend on the parameter `{$name}`

resolve_pattern_doesnt_bind_name = pattern doesn't bind `{$name}`

resolve_proc_macro_same_crate = can't use a procedural macro from the same crate that defines it
    .help = you can define integration tests in a directory named `tests`

resolve_reexport_of_crate_public =
    re-export of crate public `{$ident}`

resolve_reexport_of_private =
    re-export of private `{$ident}`

resolve_relative_2018 =
    relative paths are not supported in visibilities in 2018 edition or later
    .suggestion = try

resolve_remove_surrounding_derive =
    remove from the surrounding `derive()`

resolve_remove_unnecessary_import = remove unnecessary import

resolve_self_import_can_only_appear_once_in_the_list =
    `self` import can only appear once in an import list
    .label = can only appear once in an import list

resolve_self_import_only_in_import_list_with_non_empty_prefix =
    `self` import can only appear in an import list with a non-empty prefix
    .label = can only appear in an import list with a non-empty prefix

resolve_self_imports_only_allowed_within =
    `self` imports are only allowed within a {"{"} {"}"} list

resolve_self_imports_only_allowed_within_multipart_suggestion =
    alternatively, use the multi-path `use` syntax to import `self`

resolve_self_imports_only_allowed_within_suggestion =
    consider importing the module directly

resolve_self_in_const_generic_ty =
    cannot use `Self` in const parameter type

resolve_self_in_generic_param_default =
    generic parameters cannot use `Self` in their defaults

resolve_similarly_named_defined_here =
    similarly named {$candidate_descr} `{$candidate}` defined here

resolve_single_item_defined_here =
    {$candidate_descr} `{$candidate}` defined here

resolve_static_lifetime_is_reserved = invalid lifetime parameter name: `{$lifetime}`
    .label = 'static is a reserved lifetime name

resolve_suggestion_import_ident_directly =
    import `{$ident}` directly

resolve_suggestion_import_ident_through_reexport =
    import `{$ident}` through the re-export

resolve_tool_module_imported =
    cannot use a tool module through an import
    .note = the tool module imported here

resolve_tool_only_accepts_identifiers =
    `{$tool}` only accepts identifiers
    .label = not an identifier

resolve_tool_was_already_registered =
    tool `{$tool}` was already registered
    .label = already registered here

resolve_trait_impl_duplicate =
    duplicate definitions with name `{$name}`:
    .label = duplicate definition
    .old_span_label = previous definition here
    .trait_item_span = item in trait

resolve_trait_impl_mismatch =
    item `{$name}` is an associated {$kind}, which doesn't match its trait `{$trait_path}`
    .label = does not match trait
    .trait_impl_mismatch_label_item = item in trait
resolve_try_using_similarly_named_label =
    try using similarly named label

resolve_type_not_member_of_trait =
    type `{$type_}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_type_param_in_enum_discriminant =
    type parameters may not be used in enum discriminant values

resolve_type_param_in_non_trivial_anon_const =
    type parameters may not be used in const expressions

resolve_undeclared_label =
    use of undeclared label `{$name}`
    .label = undeclared label `{$name}`

resolve_underscore_lifetime_is_reserved = `'_` cannot be used here
    .label = `'_` is a reserved lifetime name

resolve_unexpected_res_change_ty_to_const_param_sugg =
    you might have meant to write a const parameter here

resolve_unexpected_res_use_at_op_in_slice_pat_with_range_sugg =
    if you meant to collect the rest of the slice in `{$ident}`, use the at operator

resolve_unnamed_crate_root_import =
    crate root imports need to be explicitly named: `use crate as name;`

resolve_unreachable_label =
    use of unreachable label `{$name}`
    .label = unreachable label `{$name}`
    .label_definition_span = unreachable label defined here
    .note = labels are unreachable through functions, closures, async blocks and modules

resolve_unreachable_label_similar_name_reachable =
    a label with a similar name is reachable

resolve_unreachable_label_similar_name_unreachable =
    a label with a similar name exists but is also unreachable

resolve_unreachable_label_suggestion_use_similarly_named =
    try using similarly named label

resolve_unreachable_label_with_similar_name_exists =
    a label with a similar name exists but is unreachable

resolve_variable_bound_with_different_mode =
    variable `{$variable_name}` is bound inconsistently across alternatives separated by `|`
    .label = bound in different ways
    .first_binding_span = first binding

resolve_variable_is_not_bound_in_all_patterns =
    variable `{$name}` is not bound in all patterns

resolve_variable_not_in_all_patterns = variable not in all patterns
