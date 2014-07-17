- Start Date: 2014-07-16
- RFC PR #:
- Rust Issue #:

# Summary

The variants of an enum are currently defined in the same namespace as the enum
itself. This RFC proposes to define variants under the enum's namespace. Since
1.0 is fast approaching, this RFC outlines a path which requires a single,
small, change before 1.0 that allows for the remainder to be implemented after
1.0 while preserving backwards compatibility.

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

The implementation is split into several stages. Only the first needs to be
implemented before 1.0.

## Before 1.0

To ensure that the future changes preserve backwards compatibility, we must
add a small restriction: an inherent `impl` for an enum may not define any
methods with the same name as any of the enum's variants. For example, both
methods defined for `Foo` would be forbidden:
```rust
enum Foo {
    Bar,
    Baz,
}

impl Foo {
    fn Bar(&self) {}
    fn Baz() {}
}
```
This should not affect much if any code in practice, as Rust's naming
conventions state that variants should be capitalized and method names should
not.

## With other post-1.0 work

Part of the UFCS work will (or at least discussed) allowing this:
```
trait Foo {
    fn Baz();
}

enum Bar {
    Baz,
}

impl Foo for Bar {
    fn Baz() {}
}

fn main() {
    Bar::Baz(); // instead of Foo::<for Bar>::Baz()
}
```
If this is implemented before namespaced enums, we must add a restriction,
similar to the one for intrinsic methods, that prohibits something like
`Foo::Baz` from being called as in the above example. Note that the trait may
be implemented for `Bar` without any problems, and the method may be called via
the long form. It's just `Bar::Baz()` that must be disallowed.

## Later

The compiler's resolve stage will be altered in two ways. It will place an
enum's variants into the enum's sub-namespace in a similar way to methods
defined in inherent `impl`s. Note that there is one key difference between
those methods and the variants: the methods cannot be `use`d, while the
variants can be.

In addition, when searching for a name in some namespace and the name is not
present, the compiler will search all enums defined in that namespace. If
exactly one variant with the proper name is found, the name will resolve to
that variant.  If there are multiple variants with the same name, an error will
be raised (It isn't possible to end up in this situation in code written before
this is implemented, but will be possible in code written after). This ensures
that the namespacing change is fully backwards compatible. This behavior is
more conservative than it needs to be - it could try to use type inference to
determine which of the multiple variants should be selected, for example.
However, the fallback is designed to be just that - a fallback to make sure
that code does not break in the change. It is not intended to be depended on by
code written after the change has been made.

### Examples

```rust
enum Foo {
    Bar,
    Baz,
    Buz,
}

fn ok(f: Foo) {
    use Foo::Buz;

    match f {
        Bar => {} // backwards compatibility fallback
        Foo::Baz => {}
        Buz => {}
    }
}
```

```rust
enum Fruit {
    Apple,
    Orange,
}

enum Color {
    Red,
    Orange,
}

fn broken(f: Fruit) {
    match f {
        Apple => {} // fallback
        Orange => {} // ~ ERROR ambiguous reference to a variant
    }
}

fn ok(f: Fruit) {
    use Fruit::{Apple, Orange};
    match f {
        Apple => {}
        Orange => {}
    }
}
```

```rust
enum Foo {
    Bar,
    Baz
}

struct Bar;

fn broken(f: Foo) {
    match f {
        Bar => {} // ~ ERROR expected a `Foo` but found type `Bar`
        Baz => {}
    }
}
```

In addition, we can add a default-allow lint that identifies use of the
compatibility fallback. This will allow developers to minimize the chance of
breakage down the line if we decide to remove this fallback in Rust 2.0 as well
as avoid code that will presumably end up being considered bad style. Note that
this would not trigger if the variants are explicitly reexported in the module,
so enums that are designed to be used in the current manner will not be
affected.

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
In the Rust standard library, this can be used to update libraries to a cleaner
API without breaking backwards compatibility if the library changes are made in
the same release as the language change.

# Drawbacks

This will cause a period of uncertainty in libraries, where libraries that
aren't update become unidiomatic and libraries that are updated accumulate
cruft if they attempt to maintain backwards compatibility.

The compatibility fallback logic will add complexity to resolve, and with
increased complexity comes an increased chance of bugs.

# Alternatives

We can push to switch entirely to namespaced enums before 1.0. This will allow
us to avoid the compatibility fallback and minimize library cruft, at the cost
of adding a significant amount of work to the 1.0 docket. If we decide to go
down this route, we could use the same implementation strategy laid out here to
avoid breaking the entire world at once - implement the fallback and lint at a
default (or forced) `warn` level and then remove both after some period of
time.

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
