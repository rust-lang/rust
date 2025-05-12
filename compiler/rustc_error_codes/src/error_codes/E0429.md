The `self` keyword cannot appear alone as the last segment in a `use`
declaration.

Erroneous code example:

```compile_fail,E0429
use std::fmt::self; // error: `self` imports are only allowed within a { } list
```

To use a namespace itself in addition to some of its members, `self` may appear
as part of a brace-enclosed list of imports:

```
use std::fmt::{self, Debug};
```

If you only want to import the namespace, do so directly:

```
use std::fmt;
```
