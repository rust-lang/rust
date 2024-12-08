# Deprecated features

## Passes

Rustdoc has a concept called "passes". These are transformations that
`rustdoc` runs on your documentation before producing its final output.

Customizing passes is **deprecated**. The available passes are not considered stable and may
change in any release.

In the past the most common use case for customizing passes was to omit the `strip-private` pass.
You can do this more easily, and without risk of the pass being changed, by passing
[`--document-private-items`](command-line-arguments.md#--document-private-items-show-items-that-are-not-public).
