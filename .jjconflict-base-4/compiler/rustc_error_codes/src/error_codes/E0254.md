Attempt was made to import an item whereas an extern crate with this name has
already been imported.

Erroneous code example:

```compile_fail,E0254
extern crate core;

mod foo {
    pub trait core {
        fn do_something();
    }
}

use foo::core;  // error: an extern crate named `core` has already
                //        been imported in this module

fn main() {}
```

To fix this issue, you have to rename at least one of the two imports.
Example:

```
extern crate core as libcore; // ok!

mod foo {
    pub trait core {
        fn do_something();
    }
}

use foo::core;

fn main() {}
```
