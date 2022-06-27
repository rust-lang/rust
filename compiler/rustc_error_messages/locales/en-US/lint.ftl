lint-array-into-iter =
    this method call resolves to `<&{$target} as IntoIterator>::into_iter` (due to backwards compatibility), but will resolve to <{$target} as IntoIterator>::into_iter in Rust 2021
    .use-iter-suggestion = use `.iter()` instead of `.into_iter()` to avoid ambiguity
    .remove-into-iter-suggestion = or remove `.into_iter()` to iterate by value
    .use-explicit-into-iter-suggestion =
        or use `IntoIterator::into_iter(..)` instead of `.into_iter()` to explicitly iterate by value
