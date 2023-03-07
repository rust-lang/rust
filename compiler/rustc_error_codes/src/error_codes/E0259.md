The name chosen for an external crate conflicts with another external crate
that has been imported into the current module.

Erroneous code example:

```compile_fail,E0259
extern crate core;
extern crate std as core;

fn main() {}
```

The solution is to choose a different name that doesn't conflict with any
external crate imported into the current module.

Correct example:

```
extern crate core;
extern crate std as other_name;

fn main() {}
```
