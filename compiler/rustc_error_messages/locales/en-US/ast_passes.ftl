-ast_passes_more_extern =
    for more information, visit https://doc.rust-lang.org/std/keyword.extern.html

ast_passes_forbidden_let =
    `let` expressions are not supported here
    .note = only supported directly in conditions of `if` and `while` expressions
    .not_supported_or = `||` operators are not supported in let chain expressions
    .not_supported_parentheses = `let`s wrapped in parentheses are not supported in a context with let chains

ast_passes_forbidden_let_stable =
    expected expression, found statement (`let`)
    .note = variable declaration using `let` is a statement

ast_passes_deprecated_where_clause_location =
    where clause not allowed here

ast_passes_forbidden_assoc_constraint =
    associated type bounds are not allowed within structs, enums, or unions

ast_passes_keyword_lifetime =
    lifetimes cannot use keyword names

ast_passes_invalid_label =
    invalid label name `{$name}`

ast_passes_invalid_visibility =
    unnecessary visibility qualifier
    .implied = `pub` not permitted here because it's implied
    .individual_impl_items = place qualifiers on individual impl items instead
    .individual_foreign_items = place qualifiers on individual foreign items instead

ast_passes_trait_fn_const =
    functions in traits cannot be declared const
    .label = functions in traits cannot be const

ast_passes_forbidden_lifetime_bound =
    lifetime bounds cannot be used in this context

ast_passes_forbidden_non_lifetime_param =
    only lifetime parameters can be used in this context

ast_passes_fn_param_too_many =
    function can not have more than {$max_num_args} arguments

ast_passes_fn_param_c_var_args_only =
    C-variadic function must be declared with at least one named argument

ast_passes_fn_param_c_var_args_not_last =
    `...` must be the last argument of a C-variadic function

ast_passes_fn_param_doc_comment =
    documentation comments cannot be applied to function parameters
    .label = doc comments are not allowed here

ast_passes_fn_param_forbidden_attr =
    allow, cfg, cfg_attr, deny, expect, forbid, and warn are the only allowed built-in attributes in function parameters

ast_passes_fn_param_forbidden_self =
    `self` parameter is only allowed in associated functions
    .label = not semantically valid as function parameter
    .note = associated functions are those in `impl` or `trait` definitions

ast_passes_forbidden_default =
    `default` is only allowed on items in trait impls
    .label = `default` because of this

ast_passes_assoc_const_without_body =
    associated constant in `impl` without body
    .suggestion = provide a definition for the constant

ast_passes_assoc_fn_without_body =
    associated function in `impl` without body
    .suggestion = provide a definition for the function

ast_passes_impl_assoc_ty_without_body =
    associated type in `impl` without body
    .suggestion = provide a definition for the type

ast_passes_const_without_body =
    free constant item without body
    .suggestion = provide a definition for the constant

ast_passes_static_without_body =
    free static item without body
    .suggestion = provide a definition for the static

ast_passes_ty_alias_without_body =
    free type alias without body
    .suggestion = provide a definition for the type

ast_passes_fn_without_body =
    free function without a body
    .suggestion = provide a definition for the function

ast_passes_extern_block_suggestion = if you meant to declare an externally defined function, use an `extern` block

ast_passes_ty_alias_with_bound =
    bounds on `type`s in this context have no effect

ast_passes_foreign_ty_with_bound =
    bounds on `type`s in `extern` blocks have no effect

ast_passes_impl_assoc_ty_with_bound =
    bounds on `type`s in `impl`s have no effect

ast_passes_foreign_ty_with_generic_param =
    `type`s inside `extern` blocks cannot have generic parameters
    .suggestion = remove the generic parameters
    .extern_block_label = `extern` block begins here
    .more_extern_note = {-ast_passes_more_extern}

ast_passes_foreign_ty_with_where_clause =
    `type`s inside `extern` blocks cannot have `where` clauses
    .suggestion = remove the `where` clause
    .extern_block_label = `extern` block begins here
    .more_extern_note = {-ast_passes_more_extern}

ast_passes_foreign_ty_with_body =
    incorrect `type` inside `extern` block
    .label = cannot have a body
    .body_label = the invalid body
    .extern_block_label = `extern` blocks define existing foreign types and types inside of them cannot have a body
    .more_extern_note = {-ast_passes_more_extern}

ast_passes_foreign_static_with_body =
    incorrect `static` inside `extern` block
    .label = cannot have a body
    .body_label = the invalid body
    .extern_block_label = `extern` blocks define existing foreign statics and statics inside of them cannot have a body
    .more_extern_note = {-ast_passes_more_extern}

ast_passes_foreign_fn_with_body =
    incorrect function inside `extern` block
    .label = cannot have a body
    .suggestion = remove the invalid body
    .help = you might have meant to write a function accessible through FFI, which can be done by writing `extern fn` outside of the `extern` block
    .extern_block_label = `extern` blocks define existing foreign functions and functions inside of them cannot have a body
    .more_extern_note = {-ast_passes_more_extern}

ast_passes_foreign_fn_with_qualifier =
    functions in `extern` blocks cannot have qualifiers
    .extern_block_label = in this `extern` block
    .suggestion = remove the qualifiers

ast_passes_foreign_item_non_ascii =
    items in `extern` blocks cannot use non-ascii identifiers
    .extern_block_label = in this `extern` block
    .note = this limitation may be lifted in the future; see issue #83942 <https://github.com/rust-lang/rust/issues/83942> for more information

ast_passes_forbidden_c_var_args =
    only foreign or `unsafe extern "C"` functions may be C-variadic

ast_passes_unnamed_assoc_const =
    `const` items in this context need a name
    .label = `_` is not a valid name for this `const` item

ast_passes_nomangle_item_non_ascii =
    `#[no_mangle]` requires ASCII identifier

ast_passes_mod_file_item_non_ascii =
    trying to load file for module `{$name}` with non-ascii identifier name
    .help = consider using `#[path]` attribute to specify filesystem path

ast_passes_auto_trait_with_generic_param =
    auto traits cannot have generic parameters
    .ident_label = auto trait cannot have generic parameters
    .suggestion = remove the parameters

ast_passes_auto_trait_with_super_trait_or_where_clause =
    auto traits cannot have super traits or lifetime bounds
    .ident_label = auto trait cannot have super traits or lifetime bounds
    .suggestion = remove the super traits or lifetime bounds

ast_passes_auto_trait_with_assoc_item =
    auto traits cannot have associated items
    .suggestion = remove these associated items
    .ident_label = auto trait cannot have associated items

ast_passes_generic_arg_after_constraint =
    generic arguments must come before the first constraint
    .first_constraint_label = { $constraints_len ->
        [one] constraint
       *[other] constraints
    }
    .last_arg_label = { $args_len ->
        [one] generic argument
       *[other] generic arguments
    }
    .constraints_label = {""}
    .args_label = {""}
    .suggestion = move the { $constraints_len ->
        [one] constraint
       *[other] constraints
    } after the { $args_len ->
        [one] generic argument
       *[other] generic arguments
    }

ast_passes_fn_ptr_ty_with_pat =
    patterns aren't allowed in function pointer types

ast_passes_multiple_explicit_lifetime_bound =
    only a single explicit lifetime bound is permitted

ast_passes_impl_trait_ty_in_path_param =
    `impl Trait` is not allowed in path parameters

ast_passes_impl_trait_ty_nested =
    nested `impl Trait` is not allowed
    .outer_label = outer `impl Trait`
    .nested_label = nested `impl Trait` here

ast_passes_impl_trait_ty_without_trait_bound =
    at least one trait must be specified

ast_passes_deprecated_extern_missing_abi =
    extern declarations without an explicit ABI are deprecated

ast_passes_generic_param_wrong_order =
    { $param_kind ->
        [lifetime] lifetime
       *[type_or_const] type and const
    } parameters must be declared prior to { $max_param_kind ->
        [lifetime] lifetime
       *[type_or_const] type and const
    } parameters
    .suggestion = reorder the parameters: lifetimes, then consts and types

ast_passes_obsolete_auto_trait_syntax =
    `impl Trait for .. {"{}"}` is an obsolete syntax
    .help = use `auto trait Trait {"{}"}` instead

ast_passes_unsafe_negative_impl =
    negative impls cannot be unsafe
    .negative_label = negative because of this
    .unsafe_label = unsafe because of this

ast_passes_unsafe_inherent_impl =
    inherent impls cannot be unsafe
    .unsafe_label = unsafe because of this
    .ty_label = inherent impl for this type

ast_passes_negative_inherent_impl =
    inherent impls cannot be negative
    .negative_label = negative because of this
    .ty_label = inherent impl for this type

ast_passes_default_inherent_impl =
    inherent impls cannot be `default`
    .default_label = `default` because of this
    .ty_label = inherent impl for this type
    .note = only trait implementations may be annotated with `default`

ast_passes_const_inherent_impl =
    inherent impls cannot be `const`
    .const_label = `const` because of this
    .ty_label = inherent impl for this type
    .note = only trait implementations may be annotated with `const`

ast_passes_unsafe_extern_block =
    extern block cannot be declared unsafe

ast_passes_unsafe_module =
    module cannot be declared unsafe

ast_passes_empty_union =
    unions cannot have zero fields

ast_passes_ty_alias_with_where_clause =
    where clauses are not allowed after the type for type aliases
    .note = see issue #89122 <https://github.com/rust-lang/rust/issues/89122> for more information

ast_passes_generic_param_with_default_not_trailing =
    generic parameters with a default must be trailing

ast_passes_lifetime_nested_quantification =
    nested quantification of lifetimes

ast_passes_super_trait_with_maybe =
    `?Trait` is not permitted in supertraits
    .note = traits are `?{$path_str}` by default

ast_passes_trait_object_with_maybe =
    `?Trait` is not permitted in trait object types

ast_passes_forbidden_maybe_const =
    `~const` is not allowed here
    .trait_object = trait objects cannot have `~const` trait bounds
    .closure = closures cannot have `~const` trait bounds
    .fn_not_const = this function is not `const`, so it cannot have `~const` trait bounds

ast_passes_maybe_const_with_maybe_trait =
    `~const` and `?` are mutually exclusive

ast_passes_const_async_fn =
    functions cannot be both `const` and `async`
    .const_label = `const` because of this
    .async_label = `async` because of this
    .fn_label = {""}

ast_passes_patterns_in_foreign_fns =
    patterns aren't allowed in foreign function declarations
    .label = pattern not allowed in foreign function

ast_passes_patterns_in_fns_without_body =
    patterns aren't allowed in functions without bodies
    .label = pattern not allowed in function without body

ast_passes_equality_constraint =
    equality constraints are not yet supported in `where` clauses
    .label = not supported
    .assoc_constraint_suggestion = if `{$assoc_ty}` is an associated type you're trying to set, use the associated type binding syntax
    .note = see issue #20041 <https://github.com/rust-lang/rust/issues/20041> for more information
