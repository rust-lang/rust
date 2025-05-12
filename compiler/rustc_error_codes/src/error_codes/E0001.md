#### Note: this error code is no longer emitted by the compiler.

This error suggests that the expression arm corresponding to the noted pattern
will never be reached as for all possible values of the expression being
matched, one of the preceding patterns will match.

This means that perhaps some of the preceding patterns are too general, this
one is too specific or the ordering is incorrect.

For example, the following `match` block has too many arms:

```
match Some(0) {
    Some(bar) => {/* ... */}
    x => {/* ... */} // This handles the `None` case
    _ => {/* ... */} // All possible cases have already been handled
}
```

`match` blocks have their patterns matched in order, so, for example, putting
a wildcard arm above a more specific arm will make the latter arm irrelevant.

Ensure the ordering of the match arm is correct and remove any superfluous
arms.
