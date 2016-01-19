- Feature Name: `union`
- Start Date: 2015-12-29
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Provide native support for C-compatible unions, defined via a built-in syntax
macro `union!`.

# Motivation
[motivation]: #motivation

Many FFI interfaces include unions.  Rust does not currently have any native
representation for unions, so users of these FFI interfaces must define
multiple structs and transmute between them via `std::mem::transmute`.  The
resulting FFI code must carefully understand platform-specific size and
alignment requirements for structure fields.  Such code has little in common
with how a C client would invoke the same interfaces.

Introducing native syntax for unions makes many FFI interfaces much simpler and
less error-prone to write, simplifying the creation of bindings to native
libraries, and enriching the Rust/Cargo ecosystem.

A native union mechanism would also simplify Rust implementations of
space-efficient or cache-efficient structures relying on value representation,
such as machine-word-sized unions using the least-significant bits of aligned
pointers to distinguish cases.

The syntax proposed here avoids reserving a new keyword (such as `union`), and
thus will not break any existing code.  This syntax also avoids adding a pragma
to some existing keyword that doesn't quite fit, such as `struct` or `enum`,
which avoids attaching any of the semantic significance of those keywords to
this new construct.  Rust does not produce an error or warning about the
redefinition of a macro already defined in the standard library, so the
proposed syntax will not even break code that currently defines a macro named
`union!`.

To preserve memory safety, accesses to union fields may only occur in unsafe
code.  Commonly, code using unions will provide safe wrappers around unsafe
union field accesses.

# Detailed design
[design]: #detailed-design

## Declaring a union type

A union declaration uses the same field declaration syntax as a struct
declaration, except with `union!` in place of `struct`.

```rust
union! MyUnion {
    f1: u32,
    f2: f32,
}
```

`union!` implies `#[repr(C)]` as the default representation.

## Instantiating a union

A union instantiation uses the same syntax as a struct instantiation, except
that it must specify exactly one field:

```rust
let u = MyUnion { f1: 1 };
```

Specifying multiple fields in a union instantiation results in a compiler
error.

Safe code may instantiate a union, as no unsafe behavior can occur until
accessing a field of the union.  Code that wishes to maintain invariants about
the union fields should make the union fields private and provide public
functions that maintain the invariants.

## Reading fields

Unsafe code may read from union fields, using the same dotted syntax as a
struct:

```rust
fn f(u: MyUnion) -> f32 {
    unsafe { u.f2 }
}
```

## Writing fields

Unsafe code may write to fields in a mutable union, using the same syntax as a
struct:

```rust
fn f(u: &mut MyUnion) {
    unsafe {
        u.f1 = 2;
    }
}
```

If a union contains multiple fields of different sizes, assigning to a field
smaller than the entire union must not change the memory of the union outside
that field.

## Pattern matching

Unsafe code may pattern match on union fields, using the same syntax as a
struct, without the requirement to mention every field of the union in a match
or use `..`:

```rust
fn f(u: MyUnion) {
    unsafe {
        match u {
            MyUnion { f1: 10 } => { println!("ten"); }
            MyUnion { f2 } => { println!("{}", f2); }
        }
    }
}
```

Matching a specific value from a union field makes a refutable pattern; naming
a union field without matching a specific value makes an irrefutable pattern.
Both require unsafe code.

Pattern matching may match a union as a field of a larger structure.  In
particular, when using a Rust union to implement a C tagged union via FFI, this
allows matching on the tag and the corresponding field simultaneously:

```rust
#[repr(u32)]
enum Tag { I, F }

union! U {
    i: i32,
    f: f32,
}

#[repr(C)]
struct Value {
    tag: Tag,
    u: U,
}

fn is_zero(v: Value) -> bool {
    unsafe {
        match v {
            Value { tag: I, u: U { i: 0 } } => true,
            Value { tag: F, u: U { f: 0.0 } } => true,
            _ => false,
        }
    }
}
```

Note that a pattern match on a union field that has a smaller size than the
entire union must not make any assumptions about the value of the union's
memory outside that field.

## Borrowing union fields

Unsafe code may borrow a reference to a field of a union; doing so borrows the
entire union, such that any borrow conflicting with a borrow of the union
(including a borrow of another union field or a borrow of a structure
containing the union) will produce an error.

```rust
union! U {
    f1: u32,
    f2: f32,
}

#[test]
fn test() {
    let mut u = U { f1: 1 };
    unsafe {
        let b1 = &mut u.f1;
	// let b2 = &mut u.f2; // This would produce an error
        *b1 = 5;
    }
    unsafe {
        assert_eq!(u.f1, 5);
    }
}
```

Simultaneous borrows of multiple fields of a struct contained within a union do
not conflict:

```rust
struct S {
    x: u32,
    y: u32,
}

union! U {
    s: S,
    both: u64,
}

#[test]
fn test() {
    let mut u = U { s: S { x: 1, y: 2 } };
    unsafe {
        let bx = &mut u.s.x;
        // let bboth = &mut u.both; // This would fail
        let by = &mut u.s.y;
        *bx = 5;
        *by = 10;
    }
    unsafe {
        assert_eq!(u.s.x, 5);
        assert_eq!(u.s.y, 10);
    }
}
```

## Union and field visibility

The `pub` keyword works on the union and on its fields, as with a struct.  The
union and its fields default to private.  Using a private field in a union
instantiation, field access, or pattern match produces an error.

## Uninitialized unions

The compiler should consider a union uninitialized if declared without an
initializer.  However, providing a field during instantiation, or assigning to
a field, should cause the compiler to treat the entire union as initialized.

## Unions and traits

A union may have trait implementations, using the same syntax as a struct.

The compiler should provide a lint if a union field has a type that implements
the `Drop` trait.  The compiler may optionally provide a pragma to disable that
lint, for code that intentionally stores a type with Drop in a union.  The
compiler must never implicitly generate a Drop implementation for the union
itself, though Rust code may explicitly implement Drop for a union type.

## Unions and undefined behavior

Rust code must not use unions to invoke [undefined
behavior](https://doc.rust-lang.org/nightly/reference.html#behavior-considered-undefined).
In particular, Rust code must not use unions to break the pointer aliasing
rules with raw pointers, or access a field containing a primitive type with an
invalid value.

## Union size and alignment

A union must have the same size and alignment as an equivalent C union
declaration for the target platform.  Typically, a union would have the maximum
size of any of its fields, and the maximum alignment of any of its fields.
Note that those maximums may come from different fields; for instance:

```rust
union! U {
    f1: u16,
    f2: [u8; 4],
}

#[test]
fn test() {
    assert_eq!(std::mem::size_of<U>(), 4);
    assert_eq!(std::mem::align_of<U>(), 2);
}
```

# Drawbacks
[drawbacks]: #drawbacks

Adding a new type of data structure would increase the complexity of the
language and the compiler implementation, albeit marginally.  However, this
change seems likely to provide a net reduction in the quantity and complexity
of unsafe code.

# Alternatives
[alternatives]: #alternatives

This proposal has a substantial history, with many variants and alternatives
prior to the current macro-based syntax.  Thanks to many people in the Rust
community for helping to refine this RFC.

As an alternative to the macro syntax, Rust could support unions via a new
keyword instead.  However, any introduction of a new keyword will necessarily
break some code that previously compiled, such as code using the keyword as an
identifier.  Using `union` as the keyword would break the substantial volume of
existing Rust code using `union` for other purposes, including [multiple
functions in the standard
library](https://doc.rust-lang.org/std/?search=union).  Another keyword such as
`untagged_union` would reduce the likelihood of breaking code in practice;
however, in the absence of an explicit policy for introducing new keywords,
this RFC opts to not propose a new keyword.

To avoid breakage caused by a new reserved keyword, Rust could use a compound
keyword like `unsafe union` (currently not legal syntax in any context), while
not reserving `union` on its own as a keyword, to avoid breaking use of `union`
as an identifier.  This provides equally reasonable syntax, but potentially
introduces more complexity in the Rust parser.

In the absence of a new keyword, since unions represent unsafe, untagged sum
types, and enum represents safe, tagged sum types, Rust could base unions on
enum instead.  The [unsafe enum](https://github.com/rust-lang/rfcs/pull/724)
proposal took this approach, introducing unsafe, untagged enums, identified
with `unsafe enum`; further discussion around that proposal led to the
suggestion of extending it with struct-like field access syntax.  Such a
proposal would similarly eliminate explicit use of `std::mem::transmute`, and
avoid the need to handle platform-specific size and alignment requirements for
fields.

The standard pattern-matching syntax of enums would make field accesses
significantly more verbose than struct-like syntax, and in particular would
typically require more code inside unsafe blocks.  Adding struct-like field
access syntax would avoid that; however, pairing an enum-like definition with
struct-like usage seems confusing for developers.  A declaration using `enum`
leads users to expect enum-like syntax; a new construct distinct from both
`enum` and `struct` avoids leading users to expect any particular syntax or
semantics.  Furthermore, developers used to C unions will expect struct-like
field access for unions.

Since this proposal uses struct-like syntax for declaration, initialization,
pattern matching, and field access, the original version of this RFC used a
pragma modifying the `struct` keyword: `#[repr(union)] struct`.  However, while
the proposed unions match struct syntax, they do not share the semantics of
struct; most notably, unions represent a sum type, while structs represent a
product type.  The new construct `union!` avoids the semantics attached to
existing keywords.

In the absence of any native support for unions, developers of existing Rust
code have resorted to either complex platform-specific transmute code, or
complex union-definition macros.  In the latter case, such macros make field
accesses and pattern matching look more cumbersome and less structure-like, and
still require detailed platform-specific knowledge of structure layout and
field sizes.  The implementation and use of such macros provides strong
motivation to seek a better solution, and indeed existing writers and users of
such macros have specifically requested native syntax in Rust.

Finally, to call more attention to reads and writes of union fields, field
access could use a new access operator, rather than the same `.` operator used
for struct fields.  This would make union fields more obvious at the time of
access, rather than making them look syntactically identical to struct fields
despite the semantic difference in storage representation.  However, this does
not seem worth the additional syntactic complexity and divergence from other
languages.  Union field accesses already require unsafe blocks, which calls
attention to them.  Calls to unsafe functions use the same syntax as calls to
safe functions.

# Unresolved questions
[unresolved]: #unresolved-questions

Can the borrow checker support the rule that "simultaneous borrows of multiple
fields of a struct contained within a union do not conflict"?  If not, omitting
that rule would only marginally increase the verbosity of such code, by
requiring an explicit borrow of the entire struct first.

Can a pattern match match multiple fields of a union at once?  For rationale,
consider a union using the low bits of an aligned pointer as a tag; a pattern
match may match the tag using one field and a value identified by that tag
using another field.  However, if this complicates the implementation, omitting
it would not significantly complicate code using unions.

C APIs using unions often also make use of anonymous unions and anonymous
structs.  For instance, a union may contain anonymous structs to define
non-overlapping fields, and a struct may contain an anonymous union to define
overlapping fields.  This RFC does not define anonymous unions or structs, but
a subsequent RFC may wish to do so.
