- Start Date: 2014-07-16
- RFC PR #: https://github.com/rust-lang/rfcs/pull/390
- Rust Issue #: https://github.com/rust-lang/rust/issues/18478

# Summary

The variants of an enum are currently defined in the same namespace as the enum
itself. This RFC proposes to define variants under the enum's namespace.

## Note

In the rest of this RFC, *flat enums* will be used to refer to the current enum
behavior, and *namespaced enums* will be used to refer to the proposed enum
behavior.

# Motivation

Simply put, flat enums are the wrong behavior. They're inconsistent with the
rest of the language and harder to work with.

## Practicality

Some people prefer flat enums while others prefer namespaced enums. It is
trivial to emulate flat enums with namespaced enums:
```rust
pub use MyEnum::*;

pub enum MyEnum {
    Foo,
    Bar,
}
```
On the other hand, it is *impossible* to emulate namespaced enums with the
current enum system. It would have to look something like this:
```rust
pub enum MyEnum {
    Foo,
    Bar,
}

pub mod MyEnum {
    pub use super::{Foo, Bar};
}
```
However, it is now forbidden to have a type and module with the same name in
the same namespace. This workaround was one of the rationales for the rejection
of the `enum mod` proposal previously.

Many of the variants in Rust code today are *already* effectively namespaced,
by manual name mangling. As an extreme example, consider the enums in
`syntax::ast`:
```rust
pub enum Item_ {
    ItemStatic(...),
    ItemFn(...),
    ItemMod(...),
    ItemForeignMod(...),
    ...
}

pub enum Expr_ {
    ExprBox(...),
    ExprVec(...),
    ExprCall(...),
    ...
}

...
```
These long names are unavoidable as all variants of the 47 enums in the `ast`
module are forced into the same namespace.

Going without name mangling is a risky move. Sometimes variants have to be
inconsistently mangled, as in the case of `IoErrorKind`. All variants are
un-mangled (e.g, `EndOfFile` or `ConnectionRefused`) except for one,
`OtherIoError`. Presumably, `Other` would be too confusing in isolation. One
also runs the risk of running into collisions as the library grows.

## Consistency

Flat enums are inconsistent with the rest of the language. Consider the set of
items. Some don't have their own names, such as `extern {}` blocks, so items
declared inside of them have no place to go but the enclosing namespace. Some
items do not declare any "sub-names", like `struct` definitions or statics.
Consider all other items, and how sub-names are accessed:
```rust
mod foo {
    fn bar() {}
}

foo::bar()
```

```rust
trait Foo {
    type T;

    fn bar();
}

Foo::T
Foo::bar()
```

```rust
impl Foo {
    fn bar() {}
    fn baz(&self) {}
}

Foo::bar()
Foo::baz(a_foo) // with UFCS
```

```rust
enum Foo {
    Bar,
}

Bar // ??
```

Enums are the odd one out.

Current Rustdoc output reflects this inconsistency. Pages in Rustdoc map to
namespaces. The documentation page for a module contains all names defined
in its namespace - structs, typedefs, free functions, reexports, statics,
enums, but *not* variants. Those are placed on the enum's own page, next to
the enum's inherent methods which *are* placed in the enum's namespace. In
addition, search results incorrectly display a path for variant results that
contains the enum itself, such as `std::option::Option::None`.  These issues
can of course be fixed, but that will require adding more special cases to work
around the inconsistent behavior of enums.

## Usability

This inconsistency makes it harder to work with enums compared to other items.

There are two competing forces affecting the design of libraries. On one hand,
the author wants to limit the size of individual files by breaking the crate
up into multiple modules. On the other hand, the author does not necessarily
want to expose that module structure to consumers of the library, as overly
deep namespace hierarchies are hard to work with. A common solution is to use
private modules with public reexports:
```rust
// lib.rs
pub use inner_stuff::{MyType, MyTrait};

mod inner_stuff;

// a lot of code
```
```rust
// inner_stuff.rs
pub struct MyType { ... }

pub trait MyTrait { ... }

// a lot of code
```
This strategy does not work for flat enums in general. It is not all that
uncommon for an enum to have *many* variants - for example, take
[`rust-postgres`'s `SqlState`
enum](http://www.rust-ci.org/sfackler/rust-postgres/doc/postgres/error/enum.PostgresSqlState.html),
which contains 232 variants. It would be ridiculous to `pub use` all of them!
With namespaced enums, this kind of reexport becomes a simple `pub use` of the
enum itself.

Sometimes a developer wants to use many variants of an enum in an "unqualified"
manner, without qualification by the containing module (with flat enums) or
enum (with namespaced enums). This is especially common for private, internal
enums within a crate. With flat enums, this is trivial within the module in
which the enum is defined, but very painful anywhere else, as it requires each
variant to be `use`d individually, which can get *extremely* verbose. For
example, take this [from
`rust-postgres`](https://github.com/sfackler/rust-postgres/blob/557a159a8a4a8e33333b06ad2722b1322e95566c/src/lib.rs#L97-L136):
```rust
use message::{AuthenticationCleartextPassword,
              AuthenticationGSS,
              AuthenticationKerberosV5,
              AuthenticationMD5Password,
              AuthenticationOk,
              AuthenticationSCMCredential,
              AuthenticationSSPI,
              BackendKeyData,
              BackendMessage,
              BindComplete,
              CommandComplete,
              CopyInResponse,
              DataRow,
              EmptyQueryResponse,
              ErrorResponse,
              NoData,
              NoticeResponse,
              NotificationResponse,
              ParameterDescription,
              ParameterStatus,
              ParseComplete,
              PortalSuspended,
              ReadyForQuery,
              RowDescription,
              RowDescriptionEntry};
use message::{Bind,
              CancelRequest,
              Close,
              CopyData,
              CopyDone,
              CopyFail,
              Describe,
              Execute,
              FrontendMessage,
              Parse,
              PasswordMessage,
              Query,
              StartupMessage,
              Sync,
              Terminate};
use message::{WriteMessage, ReadMessage};
```
A glob import can't be used because it would pull in other, unwanted names from
the `message` module. With namespaced enums, this becomes far simpler:
```rust
use messages::BackendMessage::*;
use messages::FrontendMessage::*;
use messages::{FrontendMessage, BackendMessage, WriteMessage, ReadMessage};
```

# Detailed design

The compiler's resolve stage will be altered to place the value and type
definitions for variants in their enum's module, just as methods of inherent
impls are. Variants will be handled differently than those methods are,
however. Methods cannot currently be directly imported via `use`, while
variants will be. The determination of importability currently happens at the
module level. This logic will be adjusted to move that determination to the
definition level. Specifically, each definition will track its "importability",
just as it currently tracks its "publicness". All definitions will be
importable except for methods in implementations and trait declarations.

The implementation will happen in two stages. In the first stage, resolve will
be altered as described above. However, variants will be defined in *both* the
flat namespace and nested namespace. This is necessary t keep the compiler
bootstrapping.

After a new stage 0 snapshot, the standard library will be ported and resolve
will be updated to remove variant definitions in the flat namespace. This will
happen as one atomic PR to keep the implementation phase as compressed as
possible. In addition, if unforseen problems arise during this set of work, we
can roll back the initial commit and put the change off until after 1.0, with
only a small pre-1.0 change required. This initial conversion will focus on
making the minimal set of changes required to port the compiler and standard
libraries by reexporting variants in the old location. Later work can alter
the APIs to take advantage of the new definition locations.

## Library changes

Library authors can use reexports to take advantage of enum namespacing without
causing too much downstream breakage:
```rust
pub enum Item {
    ItemStruct(Foo),
    ItemStatic(Bar),
}
```
can be transformed to
```rust
pub use Item::Struct as ItemStruct;
pub use Item::Static as ItemStatic;

pub enum Item {
    Struct(Foo),
    Static(Bar),
}
```
To simply keep existing code compiling, a glob reexport will suffice:
```rust
pub use Item::*;

pub enum Item {
    ItemStruct(Foo),
    ItemStatic(Bar),
}
```
Once RFC #385 is implemented, it will be possible to write a syntax extension
that will automatically generate the reexport:
```rust
#[flatten]
pub enum Item {
    ItemStruct(Foo),
    ItemStatic(Bar),
}
```

# Drawbacks

The transition period will cause enormous breakage in downstream code. It is
also a fairly large change to make to resolve, which is already a bit fragile.

# Alternatives

We can implement enum namespacing after 1.0 by adding a "fallback" case to
resolve, where variants can be referenced from their "flat" definition location
if no other definition would conflict in that namespace. In the grand scheme of
hacks to preserve backwards compatibility, this is not that bad, but still
decidedly worse than not having to worry about fallback at all.

Earlier iterations of namespaced enum proposals suggested preserving flat enums
and adding `enum mod` syntax for namespaced enums. However, variant namespacing
isn't a large enough enough difference for the additon of a second way to
define enums to hold its own weight as a language feature. In addition, it
would simply cause confusion, as library authors need to decide which one they
want to use, and library consumers need to double check which place they can
import variants from.

# Unresolved questions

A recent change placed enum variants in the type as well as the value namespace
to allow for future language expansion. This broke some code that looked like
this:
```rust
pub enum MyEnum {
    Foo(Foo),
    Bar(Bar),
}

pub struct Foo { ... }
pub struct Bar { ... }
```
Is it possible to make such a declaration legal in a world with namespaced
enums? The variants `Foo` and `Bar` would no longer clash with the structs
`Foo` and `Bar`, from the perspective of a consumer of this API, but the
variant declarations `Foo(Foo)` and `Bar(Bar)` are ambiguous, since the `Foo`
and `Bar` structs will be in scope inside of the `MyEnum` declaration.
