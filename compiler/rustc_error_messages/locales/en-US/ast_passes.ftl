ast_passes_forbidden_let =
    `let` expressions are not supported here
    .note = only supported directly in conditions of `if` and `while` expressions
    .not_supported_or = `||` operators are not supported in let chain expressions
    .not_supported_parentheses = `let`s wrapped in parentheses are not supported in a context with let chains

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
