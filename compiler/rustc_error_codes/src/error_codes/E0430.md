The `self` import appears more than once in the list.

Erroneous code example:

```compile_fail,E0430
use something::{self, self}; // error: `self` import can only appear once in
                             //        the list
```

Please verify you didn't misspell the import name or remove the duplicated
`self` import. Example:

```
# mod something {}
# fn main() {
use something::{self}; // ok!
# }
```
