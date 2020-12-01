An invalid name was used for a lifetime parameter.

Erroneous code example:

```compile_fail,E0262
// error, invalid lifetime parameter name `'static`
fn foo<'static>(x: &'static str) { }
```

Declaring certain lifetime names in parameters is disallowed. For example,
because the `'static` lifetime is a special built-in lifetime name denoting
the lifetime of the entire program, this is an error:
