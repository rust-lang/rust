A generic type was described using parentheses rather than angle brackets.

Erroneous code example:

```compile_fail,E0214
let v: Vec(&str) = vec!["foo"];
```

This is not currently supported: `v` should be defined as `Vec<&str>`.
Parentheses are currently only used with generic types when defining parameters
for `Fn`-family traits.

The previous code example fixed:

```
let v: Vec<&str> = vec!["foo"];
```
