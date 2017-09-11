- Feature Name: use_nested_groups
- Start Date: 2017-08-25
- RFC PR: https://github.com/rust-lang/rfcs/pull/2128
- Rust Issue: https://github.com/rust-lang/rust/issues/44494

# Summary
[summary]: #summary

Permit nested `{}` groups in imports.  
Permit `*` in `{}` groups in imports.

```rust
use syntax::{
    tokenstream::TokenTree, // >1 segments
    ext::base::{ExtCtxt, MacResult, DummyResult, MacEager}, // nested braces
    ext::build::AstBuilder,
    ext::quote::rt::Span,
};

use syntax::ast::{self, *}; // * in braces

use rustc::mir::{*, transform::{MirPass, MirSource}}; // both * and nested braces
```

# Motivation
[motivation]: #motivation

The motivation is ergonomics.
Prefixes are often shared among imports, especially if many imports
import names from the same crate. With this nested grouping it's more often
possible to merge common import prefixes and write them once instead of writing
them multiple times.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Several `use` items with common prefix can be merged into one `use` item,
in which the prefix is written once and all the suffixes are listed inside
curly braces `{}`.  
All kinds of suffixes can be listed inside curly braces, including globs `*` and
"subtrees" with their own curly braces.

```rust
// BEFORE
use syntax::tokenstream::TokenTree;
use syntax::ext::base::{ExtCtxt, MacResult, DummyResult, MacEager};
use syntax::ext::build::AstBuilder,
use syntax::ext::quote::rt::Span,

use syntax::ast;
use syntax::ast::*;

use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource};

// AFTER
use syntax::{
    // paths with >1 segments are permitted inside braces
    tokenstream::TokenTree,
    // nested braces are permitted as well
    ext::base::{ExtCtxt, MacResult, DummyResult, MacEager},
    ext::build::AstBuilder,
    ext::quote::rt::Span,
};

// `*` can be listed in braces too
use syntax::ast::{self, *};

// both `*` and nested braces
use rustc::mir::{*, transform::{MirPass, MirSource}};

// the prefix can be empty
use {
    syntax::ast::*;
    rustc::mir::*;
};

// `pub` imports can use this syntax as well
pub use self::Visibility::{self, Public, Inherited};
```

A `use` item with merged prefixes behaves identically to several `use` items
with all the prefixes "unmerged".

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Syntax:
```
IMPORT = ATTRS VISIBILITY `use` [`::`] IMPORT_TREE `;`

IMPORT_TREE = `*` |
              REL_MOD_PATH `::` `*` |
              `{` IMPORT_TREE_LIST `}` |
              REL_MOD_PATH `::` `{` IMPORT_TREE_LIST `}` |
              REL_MOD_PATH [`as` IDENT]

IMPORT_TREE_LIST = Ø | (IMPORT_TREE `,`)* IMPORT_TREE [`,`]

REL_MOD_PATH = (IDENT `::`)* IDENT
```

Resolution:  
First the import tree is prefixed with `::`, unless it already starts with
`::`, `self` or `super`.  
Then resolution is performed as if the whole import tree were flattened, except
that `{self}`/`{self as name}` are processed specially because `a::b::self`
is illegal.

```rust
use a::{
    b::{self as s, c, d as e},
    f::*,
    g::h as i,
    *,
};

=>

use ::a::b as s;
use ::a::b::c;
use ::a::b::d as e;
use ::a::f::*;
use ::a::g::h as i;
use ::a::*;
```

Various corner cases are resolved naturally through desugaring
```rust
use an::{*, *}; // Use an owl!

=>

use an::*;
use an::*; // Legal, but reported as unused by `unused_imports` lint.
```

# Relationships with other proposal

This RFC is an incremental improvement largely independent from other
import-related proposals, but it can have effect on some other RFCs.

Some RFCs propose new syntaxes for absolute paths in the current crate
and paths from other crates. Some arguments in those proposals are based on
usage statistics - "imports from other crates are more common" or "imports from
the current crate are more common". More common imports are supposed to get
less verbose syntax.

This RFC removes the these statistics from the equation by reducing verbosity
for all imports with common prefix.  
For example, the difference in verbosity between `A`, `B` and
`C` is minimal and doesn't depend on the number of imports.
```rust
// A
use extern::{
    a::b::c,
    d::e::f,
    g::h::i,
};
// B
use crate::{
    a::b::c,
    d::e::f,
    g::h::i,
};
// C
use {
    a::b::c,
    d::e::f,
    g::h::i,
};
```

# Drawbacks
[drawbacks]: #drawbacks

The feature encourages (but not requires) multi-line formatting of a single
import
```rust
use prefix::{
    MyName,
    x::YourName,
    y::Surname,
};
```
With this formatting it becomes harder to grep for `use.*MyName`.

# Rationale and Alternatives
[alternatives]: #alternatives

Status quo is always an alternative.

# Unresolved questions
[unresolved]: #unresolved-questions

None so far.
