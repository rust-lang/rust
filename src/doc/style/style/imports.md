% Imports [FIXME: needs RFC]

The imports of a crate/module should consist of the following
sections, in order, with a blank space between each:

* `extern crate` directives
* external `use` imports
* local `use` imports
* `pub use` imports

For example:

```rust
// Crates.
extern crate getopts;
extern crate mylib;

// Standard library imports.
use getopts::{optopt, getopts};
use std::os;

// Import from a library that we wrote.
use mylib::webserver;

// Will be reexported when we import this module.
pub use self::types::Webdata;
```

### Avoid `use *`, except in tests.

Glob imports have several downsides:
* They make it harder to tell where names are bound.
* They are forwards-incompatible, since new upstream exports can clash
  with existing names.

When writing a [`test` submodule](../testing/README.md), importing `super::*` is appropriate
as a convenience.

### Prefer fully importing types/traits while module-qualifying functions.

For example:

```rust
use option::Option;
use mem;

let i: int = mem::transmute(Option(0));
```

> **[FIXME]** Add rationale.
