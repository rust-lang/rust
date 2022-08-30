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

ast_passes_trait_fn_async =
    functions in traits cannot be declared `async`
    .label = `async` because of this
    .note = `async` trait functions are not currently supported
    .note2 = consider using the `async-trait` crate: https://crates.io/crates/async-trait

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

ast_passes_assoc_type_without_body =
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
    .extern_block_suggestion = if you meant to declare an externally defined function, use an `extern` block
