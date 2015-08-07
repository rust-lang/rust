- Feature Name: use_group_as
- Start Date: 2015-02-15
- RFC PR: [rust-lang/rfcs#1219](https://github.com/rust-lang/rfcs/pull/1219)
- Rust Issue: [rust-lang/rust#27578](https://github.com/rust-lang/rust/issues/27578)

# Summary

Allow renaming imports when importing a group of symbols from a module.

```rust
use std::io::{
    Error as IoError,
    Result as IoResult,
    Read,
    Write
}
```

# Motivation

The current design requires the above example to be written like this:

```rust
use std::io::Error as IoError;
use std::io::Result as IoResult;
use std::io::{Read, Write};
```

It's unfortunate to duplicate `use std::io::` on the 3 lines, and the proposed
example feels logical, and something you reach for in this instance, without
knowing for sure if it worked.

# Detailed design

The current grammar for use statements is something like:

```
  use_decl : "pub" ? "use" [ path "as" ident
                            | path_glob ] ;

  path_glob : ident [ "::" [ path_glob
                            | '*' ] ] ?
            | '{' path_item [ ',' path_item ] * '}' ;

  path_item : ident | "self" ;
```

This RFC proposes changing the grammar to something like:

```
  use_decl : "pub" ? "use" [ path [ "as" ident ] ?
                            | path_glob ] ;

  path_glob : ident [ "::" [ path_glob
                            | '*' ] ] ?
            | '{' path_item [ ',' path_item ] * '}' ;

  path_item : ident [ "as" ident] ?
            | "self" [ "as" ident];
```

The `"as" ident` part is optional in each location, and if omitted, it is expanded
to alias to the same name, e.g. `use foo::{bar}` expands to `use foo::{bar as bar}`.

This includes being able to rename `self`, such as `use std::io::{self
as stdio, Result as IoResult};`.

# Drawbacks

# Alternatives

# Unresolved Questions
