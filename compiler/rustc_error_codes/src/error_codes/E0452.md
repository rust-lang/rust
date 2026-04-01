An invalid lint attribute has been given.

Erroneous code example:

```compile_fail,E0452
#![allow(foo = "")] // error: malformed lint attribute
```

Lint attributes only accept a list of identifiers (where each identifier is a
lint name). Ensure the attribute is of this form:

```
#![allow(foo)] // ok!
// or:
#![allow(foo, foo2)] // ok!
```
