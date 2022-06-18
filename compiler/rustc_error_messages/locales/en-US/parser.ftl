parser-struct-literal-body-without-path =
    struct literal body without path
    .suggestion = you might have forgotten to add the struct literal inside the block

parser-maybe-report-ambiguous-plus =
    ambiguous `+` in a type
    .suggestion = use parentheses to disambiguate

parser-maybe-recover-from-bad-type-plus =
    expected a path on the left-hand side of `+`, not `{$ty}`

parser-add-paren = try adding parentheses

parser-forgot-paren = perhaps you forgot parentheses?

parser-expect-path = expected a path

parser-maybe-recover-from-bad-qpath-stage-2 =
    missing angle brackets in associated item path
    .suggestion = try: `{$ty}`

parser-incorrect-semicolon =
    expected item, found `;`
    .suggestion = remove this semicolon
    .help = {$name} declarations are not followed by a semicolon

parser-incorrect-use-of-await =
    incorrect use of `await`
    .parentheses-suggestion = `await` is not a method call, remove the parentheses
    .postfix-suggestion = `await` is a postfix operation

parser-in-in-typo =
    expected iterable, found keyword `in`
    .suggestion = remove the duplicated `in`
