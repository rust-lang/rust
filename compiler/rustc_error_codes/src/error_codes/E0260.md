The name for an item declaration conflicts with an external crate's name.

Erroneous code example:

```compile_fail,E0260
extern crate core;

struct core;

fn main() {}
```

There are two possible solutions:

Solution #1: Rename the item.

```
extern crate core;

struct xyz;
```

Solution #2: Import the crate with a different name.

```
extern crate core as xyz;

struct abc;
```

See the [Declaration Statements][declaration-statements] section of the
reference for more information about what constitutes an item declaration
and what does not.

[declaration-statements]: https://doc.rust-lang.org/reference/statements.html#declaration-statements
