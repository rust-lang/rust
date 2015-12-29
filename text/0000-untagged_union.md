- Feature Name: `untagged_union`
- Start Date: 2015-12-29
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Provide native support for C-compatible unions, defined via a new keyword
`untagged_union`.

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

The syntax proposed here avoids reserving `union` as the new keyword, as
existing Rust code already uses `union` for other purposes, including [multiple
functions in the standard
library](https://doc.rust-lang.org/std/?search=union).

To preserve memory safety, accesses to union fields may only occur in `unsafe`
code.  Commonly, code using unions will provide safe wrappers around unsafe
union field accesses.

# Detailed design
[design]: #detailed-design

## Declaring a union type

A union declaration uses the same field declaration syntax as a `struct`
declaration, except with the keyword `untagged_union` in place of `struct`:

```rust
untagged_union MyUnion {
    f1: u32,
    f2: f32,
}
```

`untagged_union` implies `#[repr(C)]` as the default representation, making
`#[repr(C)] untagged_union` permissible but redundant.

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
particular, when using an `untagged_union` to implement a C tagged union via
FFI, this allows matching on the tag and the corresponding field
simultaneously:

```rust
#[repr(u32)]
enum Tag { I, F }

untagged_union U {
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
untagged_union U {
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

untagged_union U {
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

The compiler should warn if a union field has a type that implements the `Drop`
trait.

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
untagged_union U {
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

- Don't do anything, and leave users of FFI interfaces with unions to continue
  writing complex platform-specific transmute code.
- Create macros to define unions and access their fields.  However, such macros
  make field accesses and pattern matching look more cumbersome and less
  structure-like.  The implementation and use of such macros provides strong
  motivation to seek a better solution, and indeed existing writers and users
  of such macros have specifically requested native syntax in Rust.
- Define unions without a new keyword `untagged_union`, such as via
  `#[repr(union)] struct`.  This would avoid any possibility of breaking
  existing code that uses the keyword, but would make declarations more
  verbose, and introduce potential confusion with `struct` (or whatever
  existing construct the `#[repr(union)]` attribute modifies).
- Use a compound keyword like `unsafe union`, while not reserving `union` on
  its own as a keyword, to avoid breaking use of `union` as an identifier.
  Potentially more appealing syntax, if the Rust parser can support it.
- Use a new operator to access union fields, rather than the same `.` operator
  used for struct fields.  This would make union fields more obvious at the
  time of access, rather than making them look syntactically identical to
  struct fields despite the semantic difference in storage representation.
- The [unsafe enum](https://github.com/rust-lang/rfcs/pull/724) proposal:
  introduce untagged enums, identified with `unsafe enum`.  Pattern-matching
  syntax would make field accesses significantly more verbose than structure
  field syntax.
- The [unsafe enum](https://github.com/rust-lang/rfcs/pull/724) proposal with
  the addition of struct-like field access syntax.  The resulting field access
  syntax would look much like this proposal; however, pairing an enum-style
  definition with struct-style usage seems confusing for developers.  An
  enum-based declaration leads users to expect enum-like syntax; a new
  construct distinct from both enum and struct does not lead to such
  expectations, and developers used to C unions will expect struct-like field
  access for unions.

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
