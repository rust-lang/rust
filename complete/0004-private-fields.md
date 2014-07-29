- Start Date: 2014-03-11
- RFC PR: [rust-lang/rfcs#1](https://github.com/rust-lang/rfcs/pull/1)
- Rust Issue: [rust-lang/rust#8122](https://github.com/rust-lang/rust/issues/8122)

# Summary

This is an RFC to make all struct fields private by default. This includes both
tuple structs and structural structs.

# Motivation

Reasons for default private visibility

* Visibility is often how soundness is achieved for many types in rust. These
  types are normally wrapping unsafe behavior of an FFI type or some other
  rust-specific behavior under the hood (such as the standard `Vec` type).
  Requiring these types to opt-in to being sound is unfortunate.

* Forcing tuple struct fields to have non-overridable public visibility greatly
  reduces the utility of such types. Tuple structs cannot be used to create
  abstraction barriers as they can always be easily destructed.

* Private-by-default is more consistent with the rest of the Rust language. All
  other aspects of privacy are private-by-default except for enum variants. Enum
  variants, however, are a special case in that they are inserted into the
  parent namespace, and hence naturally inherit privacy.

* Public fields of a `struct` must be considered as part of the API of the type.
  This means that the exact definition of all structs is *by default* the API of
  the type. Structs must opt-out of this behavior if the `priv` keyword is
  required. By requiring the `pub` keyword, structs must opt-in to exposing more
  surface area to their API.

Reasons for inherited visibility (today's design)

* Public definitions like `pub struct Point { x: int, y: int }` are concise and
  easy to read.
* Private definitions certainly want private fields (to hide implementation
  details).

# Detailed design

Currently, rustc has two policies for dealing with the privacy of struct fields:

* Tuple structs have public fields by default (including "newtype structs")
* Fields of structural structs (`struct Foo { ... }`) inherit the same privacy
  of the enclosing struct.

This RFC is a proposal to unify the privacy of struct fields with the rest of
the language by making them private by default. This means that both tuple
variants and structural variants of structs would have private fields by
default. For example, the program below is accepted today, but would be rejected
with this RFC.

```rust
mod inner {
    pub struct Foo(u64);
    pub struct Bar { field: u64 }
}

fn main() {
    inner::Foo(10);
    inner::Bar { field: 10 };
}
```

### Refinements to structural structs

Public fields are quite a useful feature of the language, so syntax is required
to opt out of the private-by-default semantics. Structural structs already allow
visibility qualifiers on fields, and the `pub` qualifier would make the field
public instead of private.

Additionally, the `priv` visibility will no longer be allowed to modify struct
fields. Similarly to how a `priv fn` is a compiler error, a `priv` field will
become a compiler error.

### Refinements on tuple structs

As with their structural cousins, it's useful to have tuple structs with public
fields. This RFC will modify the tuple struct grammar to:

```ebnf
tuple_struct := 'struct' ident '(' fields ')' ';'
fields := field | field ',' fields
field := type | visibility type
```

For example, these definitions will be added to the language:

```rust
// a "newtype wrapper" struct with a private field
struct Foo(u64);

// a "newtype wrapper" struct with a public field
struct Bar(pub u64);

// a tuple struct with many fields, only the first and last of which are public
struct Baz(pub u64, u32, f32, pub int);
```

Public fields on tuple structs will maintain the semantics that they currently
have today. The structs can be constructed, destructed, and participate in
pattern matches.

Private fields on tuple structs will prevent the following behaviors:

* Private fields cannot be bound in patterns (both in irrefutable and refutable
  contexts, i.e. `let` and `match` statements).
* Private fields cannot be specified outside of the defining module when
  constructing a tuple struct.

These semantics are intended to closely mirror the behavior of private fields
for structural structs.

### Statistics gathered

A brief survey was performed over the entire `mozilla/rust` repository to gather
these statistics. While not representative of all projects, this repository
should give a good indication of what most structs look like in the real world.
The repository has both libraries (`libstd`) as well as libraries which don't
care much about privacy (`librustc`).

These numbers tally up all structs from all locations, and only take into
account structural structs, not tuple structs.

|                       | Inherited privacy | Private-by-default |
|-----------------------|------------------:|-------------------:|
| Private fields        |              1418 |               1852 |
| Public fields         |              2036 |               1602 |
| All-private structs   |      551 (52.23%) |       671 (63.60%) |
| All-public structs    |      468 (44.36%) |       352 (33.36%) |
| Mixed privacy structs |       36 ( 3.41%) |        32 ( 3.03%) |

The numbers clearly show that the predominant pattern is to have all-private
structs, and that there are many public fields today which can be private (and
perhaps should!). Additionally, there is on the order of 1418 instances of the
word `priv` today, when in theory there should be around `1852`. With this RFC,
there would need to be `1602` instances of the word `pub`. A very large portion
of structs requiring `pub` fields are FFI structs defined in the `libc`
module.

### Impact on enums

This RFC does not impact enum variants in any way. All enum variants will
continue to inherit privacy from the outer enum type. This includes both the
fields of tuple variants as well as fields of struct variants in enums.

# Alternatives

The main alternative to this design is what is currently implemented today,
where fields inherit the privacy of the outer structure. The pros and cons of
this strategy are discussed above.

# Unresolved questions

As the above statistics show, almost all structures are either all public or all
private. This RFC provides an easy method to make struct fields all private, but
it explicitly does not provide a method to make struct fields all public. The
statistics show that `pub` will be written less often than `priv` is today, and
it's always possible to add a method to specify a struct as all-public in the
future in a backwards-compatible fashion.

That being said, it's an open question whether syntax for an "all public struct"
is necessary at this time.
