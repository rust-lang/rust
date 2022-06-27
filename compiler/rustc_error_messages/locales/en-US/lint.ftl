lint-array-into-iter =
    this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to <{$target} as IntoIterator>::into_iter in Rust 2021
    .use-iter-suggestion = use `.iter()` instead of `.into_iter()` to avoid ambiguity
    .remove-into-iter-suggestion = or remove `.into_iter()` to iterate by value
    .use-explicit-into-iter-suggestion =
        or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value

lint-enum-intrinsics-mem-discriminant =
    the return value of `mem::discriminant` is unspecified when called with a non-enum type
    .note = the argument to `discriminant` should be a reference to an enum, but it was passed a reference to a `{$ty_param}`, which is not an enum.

lint-enum-intrinsics-mem-variant =
    the return value of `mem::variant_count` is unspecified when called with a non-enum type
    .note = the type parameter of `variant_count` should be an enum, but it was instantiated with the type `{$ty_param}`, which is not an enum.

lint-expectation = this lint expectation is unfulfilled
    .note = the `unfulfilled_lint_expectations` lint can't be expected and will always produce this message

lint-hidden-unicode-codepoints = unicode codepoint changing visible direction of text present in {$label}
    .label = this {$label} contains {$count ->
        [one] an invisible
        *[other] invisible
    } unicode text flow control {$count ->
        [one] codepoint
        *[other] codepoints
    }
    .note = these kind of unicode codepoints change the way text flows on applications that support them, but can cause confusion because they change the order of characters on the screen
    .suggestion-remove = if their presence wasn't intentional, you can remove them
    .suggestion-escape = if you want to keep them but make them visible in your source code, you can escape them
    .no-suggestion-note-escape = if you want to keep them but make them visible in your source code, you can escape them: {$escaped}
