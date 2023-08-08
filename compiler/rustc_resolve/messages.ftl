resolve_accessible_unsure = not sure whether the path is accessible or not
    .note = the type may have associated items, but we are currently not checking them

resolve_add_as_non_derive =
    add as non-Derive macro
    `#[{$macro_path}]`

resolve_added_macro_use =
    have you added the `#[macro_use]` on the module/import?

resolve_ampersand_used_without_explicit_lifetime_name =
    `&` without an explicit lifetime name cannot be used here
    .note = explicit lifetime name needed here

resolve_ancestor_only =
    visibilities can only be restricted to ancestor modules

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

resolve_cannot_find_ident_in_this_scope =
    cannot find {$expected} `{$ident}` in this scope

resolve_cannot_glob_import_possible_crates =
    cannot glob-import all possible crates

resolve_cannot_use_self_type_here =
    can't use `Self` here

resolve_change_import_binding =
    you can use `as` to change the binding name of the import

resolve_consider_adding_a_derive =
    consider adding a derive

resolve_consider_adding_macro_export =
    consider adding a `#[macro_export]` to the macro in the imported module

resolve_consider_declaring_with_pub =
    consider declaring type or module `{$ident}` with `pub`

resolve_consider_marking_as_pub =
    consider marking `{$ident}` as `pub` in the imported module

resolve_const_not_member_of_trait =
    const `{$const_}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_const_param_from_outer_fn =
    const parameter from outer function

resolve_const_param_in_enum_discriminant =
    const parameters may not be used in enum discriminant values

resolve_const_param_in_non_trivial_anon_const =
    const parameters may only be used as standalone arguments, i.e. `{$name}`

resolve_const_param_in_ty_of_const_param =
    const parameters may not be used in the type of const parameters

resolve_crate_may_not_be_imported =
    `$crate` may not be imported

resolve_crate_root_imports_must_be_named_explicitly =
    crate root imports need to be explicitly named: `use crate as name;`

resolve_expected_found =
    expected module, found {$res} `{$path_str}`
    .label = not a module

resolve_explicit_unsafe_traits =
    unsafe traits like `{$ident}` should be implemented explicitly

resolve_forward_declared_generic_param =
    generic parameters with a default cannot use forward declared identifiers
    .label = defaulted generic parameters cannot be forward declared

resolve_generic_params_from_outer_function =
    can't use generic parameters from outer function
    .label = use of generic parameter from outer function
    .suggestion = try using a local generic parameter instead

resolve_glob_import_doesnt_reexport =
    glob import doesn't reexport anything because no candidate is public enough

resolve_help_try_using_local_generic_param =
    try using a local generic parameter instead

resolve_ident_bound_more_than_once_in_parameter_list =
    identifier `{$identifier}` is bound more than once in this parameter list
    .label = used as parameter more than once

resolve_ident_bound_more_than_once_in_same_pattern =
    identifier `{$identifier}` is bound more than once in the same pattern
    .label = used in a pattern more than once

resolve_imported_crate = `$crate` may not be imported

resolve_imports_cannot_refer_to =
    imports cannot refer to {$what}

resolve_indeterminate =
    cannot determine resolution for the visibility

resolve_invalid_asm_sym =
    invalid `sym` operand
    .label = is a local variable
    .help = `sym` operands must refer to either a function or a static

resolve_is_not_directly_importable =
    `{$target}` is not directly importable
    .label = cannot be imported directly

resolve_items_in_traits_are_not_importable =
    items in traits are not importable

resolve_label_with_similar_name_reachable =
    a label with a similar name is reachable

resolve_lifetime_param_in_enum_discriminant =
    lifetime parameters may not be used in enum discriminant values

resolve_lifetime_param_in_non_trivial_anon_const =
    lifetime parameters may not be used in const expressions

resolve_lifetime_param_in_ty_of_const_param =
    lifetime parameters may not be used in the type of const parameters

resolve_lowercase_self =
    attempt to use a non-constant value in a constant
    .suggestion = try using `Self`

resolve_macro_expected_found =
    expected {$expected}, found {$found} `{$macro_path}`

resolve_macro_use_extern_crate_self = `#[macro_use]` is not supported on `extern crate self`

resolve_method_not_member_of_trait =
    method `{$method}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_module_only =
    visibility must resolve to a module

resolve_name_is_already_used_as_generic_parameter =
    the name `{$name}` is already used for a generic parameter in this item's generic parameters
    .label = already used
    .first_use_of_name = first use of `{$name}`

resolve_param_in_enum_discriminant =
    generic parameters may not be used in enum discriminant values
    .label = cannot perform const operation using `{$name}`

resolve_param_in_non_trivial_anon_const =
    generic parameters may not be used in const operations
    .label = cannot perform const operation using `{$name}`

resolve_param_in_non_trivial_anon_const_help =
    use `#![feature(generic_const_exprs)]` to allow generic const expressions

resolve_param_in_ty_of_const_param =
    the type of const parameters must not depend on other generic parameters
    .label = the type must not depend on the parameter `{$name}`

resolve_parent_module_reset_for_binding =
    parent module is reset for binding

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

resolve_self_in_generic_param_default =
    generic parameters cannot use `Self` in their defaults
    .label = `Self` in generic parameter default

resolve_self_type_implicitly_declared_by_impl =
    `Self` type implicitly declared here, by this `impl`

resolve_tool_module_imported =
    cannot use a tool module through an import
    .note = the tool module imported here

resolve_trait_impl_duplicate =
    duplicate definitions with name `{$name}`:
    .label = duplicate definition
    .old_span_label = previous definition here
    .trait_item_span = item in trait

resolve_trait_impl_mismatch =
    item `{$name}` is an associated {$kind}, which doesn't match its trait `{$trait_path}`
    .label = does not match trait
    .label_trait_item = item in trait

resolve_try_adding_local_generic_param_on_method =
    try adding a local generic parameter in this method instead

resolve_try_using_local_generic_parameter =
    try using a local generic parameter instead

resolve_try_using_similarly_named_label =
    try using similarly named label

resolve_type_not_member_of_trait =
    type `{$type_}` is not a member of trait `{$trait_}`
    .label = not a member of trait `{$trait_}`

resolve_type_param_from_outer_fn =
    type parameter from outer function

resolve_type_param_in_enum_discriminant =
    type parameters may not be used in enum discriminant values

resolve_type_param_in_non_trivial_anon_const =
    type parameters may not be used in const expressions

resolve_type_param_in_ty_of_const_param =
    type parameters may not be used in the type of const parameters

resolve_undeclared_label =
    use of undeclared label `{$name}`
    .label = undeclared label `{$name}`

resolve_underscore_lifetime_name_cannot_be_used_here =
    `'_` cannot be used here
    .note = `'_` is a reserved lifetime name

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

resolve_use_a_type_here_instead =
    use a type here instead

resolve_variable_bound_with_different_mode =
    variable `{$variable_name}` is bound inconsistently across alternatives separated by `|`
    .label = bound in different ways
    .first_binding_span = first binding
