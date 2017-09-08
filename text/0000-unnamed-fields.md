- Feature Name: unnamed_fields
- Start Date: 2017-08-05
- RFC PR:
- Rust Issue:

# Summary
[summary]: #summary

Allow unnamed fields of `struct` and `union` type, contained within an outer
struct or union; the fields they contain appear directly within the containing
structure, with the use of `union` and `struct` determining which fields have
non-overlapping storage (making them usable at the same time).  This allows
grouping and laying out fields in arbitrary ways, to match C data structures
used in FFI. The C11 standard allows this, and C compilers have allowed it for
decades as an extension. This proposal allows Rust to represent such types
using the same names as the C structures, without interposing artificial field
names that will confuse users of well-established interfaces from existing
platforms.

# Motivation
[motivation]: #motivation

Numerous C interfaces follow a common pattern, consisting of a `struct`
containing discriminants and common fields, and an unnamed `union` of fields
specific to certain values of the discriminants. To group together fields used
together as part of the same variant, these interfaces also often use unnamed
`struct` types.

Thus, `struct` defines a set of fields that can appear at the same time, and
`union` defines a set of mutually exclusive overlapping fields.

This pattern appears throughout many C APIs. The Windows and POSIX APIs both
use this pattern extensively. However, Rust currently can't represent this
pattern in a straightforward way. While Rust supports structs and unions, every
such struct and union must have a field name. When creating a binding to such
an interface, whether manually or using a binding generator, the binding must
invent an artificial field name that does not appear in the original interface.

This RFC proposes a minimal mechanism to support such interfaces in Rust. This
feature exists primarily to support ergonomic FFI interfaces that match the
layout of data structures for the native platform; this RFC intentionally
limits itself to the `repr(C)` structure representation, and does not provide
support for using this feature in Rust data structures using `repr(Rust)`. As
precedent, Rust's support for variadic argument lists only permits its use on
`extern "C"` functions.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This explanation should appear after the definition of `union`, and after an
explanation of the rationale for `union` versus `enum` in Rust.

Please note that most Rust code will want to use an `enum` to define types that
contain a discriminant and various disjoint fields. The unnamed field mechanism
here exist primarily for compatibility with interfaces defined by non-Rust
languages, such as C. Types declared with this mechanism require `unsafe` code
to access.

A `struct` defines a set of fields all available at the same time, with storage
available for each. A `union` defines (in an unsafe, unchecked manner) a set of
mutually exclusive fields, with overlapping storage. Some types and interfaces
may require nesting such groupings. For instance, a `struct` may contain a set
of common fields and a `union` of fields needed for different variations of the
structure; conversely, a `union` contain a `struct` grouping together fields
needed simultaneously.

Such groupings, however, do not always have associated types and names. A
structure may contain groupings of fields where the fields have meaningful
names, but the groupings of fields do not. In this case, the structure can
contain *unnamed fields* of `struct` or `union` type, to group the fields
together, and determine which fields overlap.

As an example, when defining a `struct`, you may have a set of fields that will
never be used at the same time, so you could overlap the storage of those
fields. This pattern often occurs within C APIs, when defining an interface
similar to a Rust `enum`. You could do so by declaring a separate `union` type
and a field of that type. With the unnamed fields mechanism, you can also
define an unnamed grouping of overlapping fields inline within the `struct`,
using the `union` keyword:

```rust
struct S {
    a: u32,
    _: union {
        b: u32,
        c: f32,
    },
    d: u64,
}
```

The underscore `_` indicates the absence of a field name; the fields within the
unnamed union will appear directly with the containing structure. Given a
struct `s` of this type, code can access `s.a`, `s.d`, and either `s.b` or
`s.c`. Accesses to `a` and `d` can occur in safe code; accesses to `b` and `c`
require unsafe code, and `b` and `c` overlap, requiring care to access only the
field whose contents make sense at the time. As with any `union`, code cannot
borrow `s.b` and `s.c` simultaneously.

Conversely, sometimes when defining a `union`, you may want to group multiple
fields together and make them available simultaneously, with non-overlapping
storage. You could do so by defining a separate `struct`, and placing an
instance of that `struct` within the `union`. With the unnamed fields
mechanism, you can also define an unnamed grouping of non-overlapping fields
inline within the `union`, using the `struct` keyword:

```rust
union U {
    a: u32,
    _: struct {
        b: u16,
        c: f16,
    },
    d: f32,
}
```

Given a union `u` of this type, code can access `u.a`, or `u.d`, or both `u.b`
and `u.c`. Since all of these fields can potentially overlap with others,
accesses to any of them require unsafe code; however, `b` and `c` do not
overlap with each other. Code can borrow `u.b` and `u.c` simultaneously, but
cannot borrow any other fields at the same time.

Structs can also contain unnamed structs, and unions can contain unnamed
unions.

Unnamed fields can contain other unnamed fields. For example:

```rust
struct S {
    a: u32,
    _: union {
        b: u32,
        _: struct {
            c: u16,
            d: f16,
        },
        e: f32,
    },
    f: u64,
}
```

This structure contains six fields: `a`, `b`, `c`, `d`, `e`, and `f`. Safe code
can access fields `a` and `f`, at any time, since those fields do not lie
within a union and do not overlap with any other field. Unsafe code can access
the remaining fields. This definition effectively acts as the overlap of the
following three structures:

```rust
// variant 1
struct S {
    a: u32,
    b: u32,
    f: u64,
}

// variant 2
struct S {
    a: u32,
    c: u16,
    d: f16,
    f: u64,
}

// variant 3
struct S {
    a: u32,
    e: f32,
    f: u64,
}
```

## Mental model

In the memory layout of a structure, the alternating uses of `struct { ... }`
and `union { ... }` change the "direction" that fields are being laid out: if
you think of memory addresses as going vertically, `struct` lays out fields
vertically, in sequence, and `union` lays out fields horizontally, overlapping
with each other. The following definition:

```rust
struct S {
    a: u32,
    _: union {
        b: u32,
        _: struct {
            c: u16,
            d: f16,
        },
        e: f32,
    },
    f: u64,
}
```

corresponds to the following structure layout in memory:

```
+-----------+ 0
|     a     |
+-----------+ 4
| b | c | e |
|   +---+   | 6
|   | d |   |
+-----------+ 8
|     f     |
+-----------+ 16
```

The top-level `struct` lays out `a`, the unnamed `union`, and `f`, in
sequential order. The unnamed `union` lays out `b`, the unnamed `struct`, and
`e`, in parallel. The unnamed `struct` lays out `c` and `d` in sequential
order.

## Instantiation

Given the following declaration:

```rust
struct S {
    a: u32,
    _: union {
        b: u32,
        _: struct {
            c: u16,
            d: f16,
        },
        e: f32,
    },
    f: u64,
}
```

All of the following will instantiate a value of type `S`:

- `S { a: 1, b: 2, f: 3.0 }`
- `S { a: 1, c: 2, d: 3.0, f: 4.0 }`
- `S { a: 1, e: 2.0, f: 3.0 }`

## Representation

This feature exists to support the layout of native platform data structures.
Structures using the default `repr(Rust)` layout cannot use this feature, and
the compiler should produce an error when attempting to do so.

When using this mechanism to define a C interface, remember to use the
`repr(C)` attribute to match C's data structure layout. Any representation
attribute applied to the top-level structure also applies to every unnamed
field within that declaration. Such a structure defined with `repr(C)` will use
a representation identical to the same structure with all unnamed fields
transformed to equivalent named fields of a struct or union type with the same
fields.

Similarly, applying `repr(packed)` to the top-level data structure will also
apply it to all the contained structures.

## Derive

A `struct` or `union` containing unnamed fields may derive `Copy`, `Clone`, or
both, if all the fields it contains (including within unnamed fields) also
implement `Copy`.

A `struct` containing unnamed fields may derive `Clone` if every field
contained directly in the `struct` implements `Clone`, and every field
contained within an unnamed `union` (directly or indirectly) implements `Copy`.

## Ambiguous field names

You cannot use this feature to define multiple fields with the same name. For
instance, the following definition will produce an error:

```rust
struct S {
    a: u32,
    _: union {
        a: u32,
        b: f32,
    },
}
```

The error will identify the duplicate `a` fields as the sources of the error.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

## Parsing

Within a struct or union's fields, in place of a field name and value, allow
`_: struct { fields }` or `_: union { fields }`, where `fields` allows
everything allowed within a `struct` or `union` declaration, respectively.

The name `_` cannot currently appear as a field name, so this will not
introduce any compatibility issues with existing code. The keyword `struct`
cannot appear as a field type, making it entirely unambiguous. The contextual
keyword `union` could theoretically appear as a type name, but an open brace
cannot appear immediately after a field type, allowing disambiguation via a
single token of context (`union {`).

## Layout and Alignment

The layout and alignment of a `struct` or `union` containing unnamed
fields should look the same as if each unnamed field has a separately declared
type and a named field of that type, rather than as if the fields appeared
directly within the containing `struct` or `union`. In some cases, this may
result in different alignment.

## Simultaneous Borrows

An unnamed `struct` within a `union` should behave the same with respect to
borrows as a named and typed `struct` within a `union`, allowing borrows of
multiple fields from within the `struct`, while not permitting borrows of other
fields in the `union`.

## Visibility

Each field within an unnamed `struct` or `union` may have an attached
visibility. An unnamed field itself does not have its own visibility; all of
its fields appear directly within the containing structure, and their own
visibilities apply.

## Documentation

Public fields within an unnamed `struct` or `union` should appear in the
rustdoc documentation of the outer structure, along with any doc comment or
attribute attached to those fields. The rendering should include all unnamed
fields that contain (at any level of nesting) a public field, and should
include the `// some fields omitted` note within any `struct` or `union` that
has non-public fields, including unnamed fields.

Any unnamed field that contains only non-public fields should be omitted
entirely, rather than included with its fields omitted. Omitting an unnamed
field should trigger the `// some fields omitted` note.

# Drawbacks
[drawbacks]: #drawbacks

This introduces additional complexity in structure definitions. Strictly
speaking, C interfaces do not *require* this mechanism; any such interface
*could* define named struct or union types, and define named fields of that
type. This RFC provides a usability improvement for such interfaces.

# Rationale and Alternatives
[alternatives]: #alternatives

## Not implementing this feature at all

Choosing not to implement this feature would force binding generators (and the
authors of manual bindings) to invent new names for these groupings of fields.
Users would need to look up the names for those groupings, and would not be
able to rely on documentation for the underlying interface. Furthermore,
binding generators would not have any basis on which to generate a meaningful
name.

## Not implementable as a macro

We cannot implement this feature as a macro, because it affects the names used
to reference the fields contained within an unnamed field. A macro could
extract and define types for the unnamed fields, but that macro would have to
give a name to those unnamed fields, and accesses would have to include the
intermediate name.

## Leaving out the `_: ` in unnamed fields

Rather than declaring unnamed fields with an `_`, as in `_: union { fields }`
and `_: struct { fields }`, we could omit the field name entirely, and write
`union { fields }` and `struct { fields }` directly. This would more closely
match the C syntax. However, this does not provide as natural an extension to
support references to named structures.

## Unnamed fields with named types

In addition to allowing the inline definition of the contents of unnamed
`struct` or `union` fields, we could also allow declaring an unnamed field with
a named type.  For instance:

```rust
union U { x: i64, y: f64 }
struct S { _: U, z: usize }
```

Given these declarations, `S` would contain fields `x`, `y`, and `z`, with `x`
and `y` overlapping.

This syntax makes it possible to give a name to the intermediate type, while
still leaving the field unnamed. While C11 does not directly support inlining
of separately defined structures, compilers do support it as an extension. This
syntax would allow for the common definition of sets of fields inlined into
several structures, such as a common header.

This syntax would also support an obvious translation of inline-declared
structures with names, by moving the declaration out-of-line; a macro could
perform such a translation.

## Field aliases

Rather than introducing unnamed fields, we could introduce a mechanism to
define field aliases for a type, such that for `struct S`, `s.b` desugars to
`s.b_or_c.b`. However, such a mechanism does not seem any simpler than unnamed
fields, and would not align as well with the potential future introduction of
full anonymous structure types. Furthermore, such a mechanism would need to
allow hiding the underlying paths for portability; for example, the `siginfo_t`
type on POSIX platforms allows portable access to certain named fields, but
different platforms overlap those fields differently using unnamed unions.
Finally, such a mechanism would make it harder to create bindings for this
common pattern in C interfaces.

## Alternate syntax

Several alternative syntaxes could exist to designate the equivalent of
`struct` and `union`. Such syntaxes would declare the same underlying types.
However, inventing a novel syntax for this mechanism would make it less
familiar both to Rust users accustomed to structs and unions as well as to C
users accustomed to unnamed struct and union fields.

## Arbitrary field positioning

We could introduce a mechanism to declare arbitrarily positioned fields, such
as attributes declaring the offset of each field. The same mechanism was also
proposed in response to the original union RFC. However, as in that case, using
struct and union syntax has the advantage of allowing the compiler to implement
the appropriate positioning and alignment of fields.

## General anonymous types

In addition to introducing just this narrow mechanism for defining unnamed
fields, we could introduce a fully general mechanism for anonymous `struct` and
`union` types that can appear anywhere a type can appear, including in function
arguments and return values, named structure fields, or local variables. Such
an anonymous type mechanism would *not* replace the need for unnamed fields,
however, and vice versa. Furthermore, anonymous types would interact
extensively with far more aspects of Rust. Such a mechanism should appear in a
subsequent RFC.

This mechanism intentionally does not provide any means to reference an unnamed
field as a whole, or its type. That intentional limitation avoids allowing such
unnamed types to propagate.

# Unresolved questions
[unresolved]: #unresolved-questions

This proposal does *not* support anonymous `struct` and `union` types that can
appear anywhere a type can appear, such as in the type of an arbitrary named
field or variable. Doing so would further simplify some C interfaces, as well
as native Rust constructs.

However, such a change would also cascade into numerous other changes, such as
anonymous struct and union literals. Unlike this proposal, anonymous aggregate
types for named fields have a reasonable alternative, namely creating and using
separate types; binding generators could use that mechanism, and a macro could
allow declaring those types inline next to the fields that use them.

Furthermore, during the pre-RFC process, that portion of the proposal proved
more controversial. And such a proposal would have a much more expansive impact
on the language as a whole, by introducing a new construct that works anywhere
a type can appear. Thus, this proposal provides the minimum change necessary to
enable bindings to these types of C interfaces.

C structures can still include other constructs that Rust does not currently
represent, including bitfields, and variable-length arrays at the end of a
structure. Future RFCs may wish to introduce support for those constructs as
well. However, I do not believe it makes sense to require a solution for every
problem of interfacing with C simultaneously, nor to gate a solution for one
common issue on solutions for others.
