A literal was used in a built-in attribute that doesn't support literals.

Erroneous code example:

```compile_fail,E0565
#[repr("C")] // error: meta item in `repr` must be an identifier
struct Repr {}

fn main() {}
```

Literals in attributes are new and largely unsupported in built-in attributes.
Work to support literals where appropriate is ongoing. Try using an unquoted
name instead:

```
#[repr(C)] // ok!
struct Repr {}

fn main() {}
```
