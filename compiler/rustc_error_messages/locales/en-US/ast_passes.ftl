ast_passes_forbidden_let =
    `let` expressions are not supported here
    .note = only supported directly in conditions of `if` and `while` expressions
    .not_supported_or = `||` operators are not supported in let chain expressions
    .not_supported_parentheses = `let`s wrapped in parentheses are not supported in a context with let chains
