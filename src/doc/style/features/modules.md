% Modules

> **[FIXME]** What general guidelines should we provide for module design?

> We should discuss visibility, nesting, `mod.rs`, and any interesting patterns
> around modules.

### Headers [FIXME: needs RFC]

Organize module headers as follows:
  1. [Imports](../style/imports.md).
  1. `mod` declarations.
  1. `pub mod` declarations.

### Avoid `path` directives. [FIXME: needs RFC]

Avoid using `#[path="..."]` directives; make the file system and
module hierarchy match, instead.

### Use the module hierarchy to organize APIs into coherent sections. [FIXME]

> **[FIXME]** Flesh this out with examples; explain what a "coherent
> section" is with examples.
>
> The module hierarchy defines both the public and internal API of your module.
> Breaking related functionality into submodules makes it understandable to both
> users and contributors to the module.

### Place modules in their own file. [FIXME: needs RFC]

> **[FIXME]**
> - "<100 lines" is arbitrary, but it's a clearer recommendation
>   than "~1 page" or similar suggestions that vary by screen size, etc.

For all except very short modules (<100 lines) and [tests](../testing/README.md),
place the module `foo` in a separate file, as in:

```rust
pub mod foo;

// in foo.rs or foo/mod.rs
pub fn bar() { println!("..."); }
/* ... */
```

rather than declaring it inline:

```rust
pub mod foo {
    pub fn bar() { println!("..."); }
    /* ... */
}
```

#### Use subdirectories for modules with children. [FIXME: needs RFC]

For modules that themselves have submodules, place the module in a separate
directory (e.g., `bar/mod.rs` for a module `bar`) rather than the same directory.

Note the structure of
[`std::io`](https://doc.rust-lang.org/std/io/). Many of the submodules lack
children, like
[`io::fs`](https://doc.rust-lang.org/std/io/fs/)
and
[`io::stdio`](https://doc.rust-lang.org/std/io/stdio/).
On the other hand,
[`io::net`](https://doc.rust-lang.org/std/io/net/)
contains submodules, so it lives in a separate directory:

```
io/mod.rs
   io/extensions.rs
   io/fs.rs
   io/net/mod.rs
          io/net/addrinfo.rs
          io/net/ip.rs
          io/net/tcp.rs
          io/net/udp.rs
          io/net/unix.rs
   io/pipe.rs
   ...
```

While it is possible to define all of `io` within a single directory,
mirroring the module hierarchy in the directory structure makes
submodules of `io::net` easier to find.

### Consider top-level definitions or reexports. [FIXME: needs RFC]

For modules with submodules,
define or [reexport](https://doc.rust-lang.org/std/io/#reexports) commonly used
definitions at the top level:

* Functionality relevant to the module itself or to many of its
  children should be defined in `mod.rs`.
* Functionality specific to a submodule should live in that
  submodule. Reexport at the top level for the most important or
  common definitions.

For example,
[`IoError`](https://doc.rust-lang.org/std/io/struct.IoError.html)
is defined in `io/mod.rs`, since it pertains to the entirety of `io`,
while
[`TcpStream`](https://doc.rust-lang.org/std/io/net/tcp/struct.TcpStream.html)
is defined in `io/net/tcp.rs` and reexported in the `io` module.

### Use internal module hierarchies for organization. [FIXME: needs RFC]

> **[FIXME]**
> - Referencing internal modules from the standard library is subject to
>   becoming outdated.

Internal module hierarchies (i.e., private submodules) may be used to
hide implementation details that are not part of the module's API.

For example, in [`std::io`](https://doc.rust-lang.org/std/io/), `mod mem`
provides implementations for
[`BufReader`](https://doc.rust-lang.org/std/io/struct.BufReader.html)
and
[`BufWriter`](https://doc.rust-lang.org/std/io/struct.BufWriter.html),
but these are re-exported in `io/mod.rs` at the top level of the module:

```rust
// libstd/io/mod.rs

pub use self::mem::{MemReader, BufReader, MemWriter, BufWriter};
/* ... */
mod mem;
```

This hides the detail that there even exists a `mod mem` in `io`, and
helps keep code organized while offering freedom to change the
implementation.
