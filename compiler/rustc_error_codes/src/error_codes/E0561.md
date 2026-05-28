A non-ident or non-wildcard pattern has been used as a parameter of a function
pointer type.

Erroneous code example:

```compile_fail,E0561
type A1 = fn(mut param: u8); // error!
type A2 = fn(&param: u32); // error!
```

When using an alias over a function type, you cannot e.g. denote a parameter as
being mutable.

To fix the issue, remove patterns (`_` is allowed though). Example:

```
type A1 = fn(param: u8); // ok!
type A2 = fn(_: u32); // ok!
```

You can also omit the parameter name:

```
type A3 = fn(i16); // ok!
```
