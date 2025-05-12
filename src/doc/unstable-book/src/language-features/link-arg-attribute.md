# `link_arg_attribute`

The tracking issue for this feature is: [#99427]

------

The `link_arg_attribute` feature allows passing arguments into the linker
from inside of the source code. Order is preserved for link attributes as
they were defined on a single extern block:

```rust,no_run
#![feature(link_arg_attribute)]

#[link(kind = "link-arg", name = "--start-group")]
#[link(kind = "static", name = "c")]
#[link(kind = "static", name = "gcc")]
#[link(kind = "link-arg", name = "--end-group")]
extern "C" {}
```

[#99427]: https://github.com/rust-lang/rust/issues/99427
