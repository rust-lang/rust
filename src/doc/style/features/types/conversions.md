% Conversions between types

### Associate conversions with the most specific type involved. **[FIXME: needs RFC]**

When in doubt, prefer `to_`/`as_`/`into_` to `from_`, because they are
more ergonomic to use (and can be chained with other methods).

For many conversions between two types, one of the types is clearly more
"specific": it provides some additional invariant or interpretation that is not
present in the other type. For example, `str` is more specific than `&[u8]`,
since it is a utf-8 encoded sequence of bytes.

Conversions should live with the more specific of the involved types. Thus,
`str` provides both the `as_bytes` method and the `from_utf8` constructor for
converting to and from `&[u8]` values. Besides being intuitive, this convention
avoids polluting concrete types like `&[u8]` with endless conversion methods.

### Explicitly mark lossy conversions, or do not label them as conversions. **[FIXME: needs RFC]**

If a function's name implies that it is a conversion (prefix `from_`, `as_`,
`to_` or `into_`), but the function loses information, add a suffix `_lossy` or
otherwise indicate the lossyness. Consider avoiding the conversion name prefix.
