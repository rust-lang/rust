# Rust Codegen

The first phase in debug info generation requires Rust to inspect the MIR of the program and
communicate it to LLVM. This is primarily done in [`rustc_codegen_llvm/debuginfo`][llvm_di], though
some type-name processing exists in [`rustc_codegen_ssa/debuginfo`][ssa_di]. Rust communicates to
LLVM via the `DIBuilder` API - a thin wrapper around LLVM's internals that exists in
[rustc_llvm][rustc_llvm].

[llvm_di]: https://github.com/rust-lang/rust/tree/main/compiler/rustc_codegen_llvm/src/debuginfo
[ssa_di]: https://github.com/rust-lang/rust/tree/main/compiler/rustc_codegen_ssa/src/debuginfo
[rustc_llvm]: https://github.com/rust-lang/rust/tree/main/compiler/rustc_llvm

# Type Information

Type information typically consists of the type name, size, alignment, as well as things like
fields, generic parameters, and storage modifiers if they are relevant. Much of this work happens in
[rustc_codegen_llvm/src/debuginfo/metadata][di_metadata].

[di_metadata]: https://github.com/rust-lang/rust/blob/main/compiler/rustc_codegen_llvm/src/debuginfo/metadata.rs

It is important to keep in mind that the goal is not necessarily "represent types exactly how they
appear in Rust", rather it is to represent them in a way that allows debuggers to most accurately
reconstruct the data during debugging. This distinction is vital to understanding the core work that
occurs on this layer; many changes made here will be for the purpose of working around debugger
limitations when no other option will work.

## Quirks

Rust's generated DI nodes "pretend" to be C/C++ for both CDB and LLDB's sake. This can result in
some unintuitive and non-idiomatic debug info.

### Pointers and Reference

Wide pointers/references/`Box` are treated as a struct with 2 fields: `data_ptr` and `length`.

All non-wide pointers, references, and `Box` pointers are output as pointer nodes, and no
distinction is made between `mut` and non-`mut`. Several attempts have been made to rectify this,
but unfortunately there is not a straightforward solution. Using the `reference` DI nodes of the
respective formats has pitfalls. There is a semantic difference between C++ references and Rust
references that is unreconcilable.

>From [cppreference](https://en.cppreference.com/w/cpp/language/reference.html):
>
>References are not objects; **they do not necessarily occupy storage**, although the compiler may
>allocate storage if it is necessary to implement the desired semantics (e.g. a non-static data
>member of reference type usually increases the size of the class by the amount necessary to store
>a memory address).
>
>Because references are not objects, **there are no arrays of references, no pointers to references, and no references to references**

The current proposed solution is to simply [typedef the pointer nodes][issue_144394].

[issue_144394]: https://github.com/rust-lang/rust/pull/144394

Using the `const` qualifier to denote non-`mut` poses potential issues due to LLDB's internal
optimizations. In short, LLDB attempts to cache the child-values of variables (e.g. struct fields,
array elements) when stepping through code. A heuristic is used to determine which values are safely
cache-able, and `const` is part of that heuristic. Research has not been done into how this would
interact with things like Rust's interior mutability constructs.

### DWARF vs PDB

While most of the type information is fairly straight forward, one notable issue is the debug info
format of the target. Each format has different semantics and limitations, as such they require
slightly different debug info in some cases. This is gated by calls to
[`cpp_like_debuginfo`][cpp_like].

[cpp_like]: https://github.com/rust-lang/rust/blob/main/compiler/rustc_codegen_ssa/src/debuginfo/type_names.rs#L813

### Naming

Rust attempts to communicate type names as accurately as possible, but debuggers and debug info
formats do not always respect that.

Due to limitations in MSVC's expression parser, the following name transformations are made for PDB
debug info:

| Rust name | MSVC name |
| --- | --- |
| `&str`/`&mut str` | `ref$<str$>`/`ref_mut$<str$>` |
| `&[T]`/`&mut [T]` | `ref$<slice$<T> >`/`ref_mut$<slice$<T> >`[^1] |
| `[T; N]` | `array$<T, N>` |
| `RustEnum` | `enum2$<RustEnum>` |
| `(T1, T2)` | `tuple$<T1, T2>`|
| `*const T` | `ptr_const$<T>` |
| `*mut T` | `ptr_mut$<T>` |
| `usize` | `size_t`[^2] |
| `isize` | `ptrdiff_t`[^2] |
| `uN` | `unsigned __intN`[^2] |
| `iN` | `__intN`[^2] |
| `f32` | `float`[^2] |
| `f64` | `double`[^2] |
| `f128` | `fp128`[^2] |

[^1]: MSVC's expression parser will treat `>>` as a right shift. It is necessary to separate
consecutive `>`'s with a space (`> >`) in type names.

[^2]: While these type names are generated as part of the debug info node (which is then wrapped in
a typedef node with the Rust name), once the LLVM-IR node is converted to a CodeView node, the type
name information is lost. This is because CodeView has special shorthand nodes for primitive types,
and those shorthand nodes to not have a "name" field.

### Generics

Rust outputs generic *type* information (`T` in `ArrayVec<T, N: usize>`), but not generic *value*
information (`N` in `ArrayVec<T, N: usize>`).

CodeView does not have a leaf node for generics/C++ templates, so all generic information is lost
when generating PDB debug info. There are workarounds that allow the debugger to retrieve the
generic arguments via the type name, but it is fragile solution at best. Efforts are being made to
contact Microsoft to correct this deficiency, and/or to use one of the unused CodeView node types as
a suitable equivalent.

### Type aliases

Rust outputs typedef nodes in several cases to help account for debugger limitiations, but it does
not currently output nodes for [type aliases in the source code][type_aliases].

[type_aliases]: https://doc.rust-lang.org/reference/items/type-aliases.html

### Enums

Enum DI nodes are generated in [rustc_codegen_llvm/src/debuginfo/metadata/enums][di_metadata_enums]

[di_metadata_enums]: https://github.com/rust-lang/rust/tree/main/compiler/rustc_codegen_llvm/src/debuginfo/metadata/enums

#### DWARF

DWARF has a dedicated node for discriminated unions: `DW_TAG_variant`. It is a container that
references `DW_TAG_variant_part` nodes that may or may not contain a discriminant value. The
hierarchy looks as follows:

```txt
DW_TAG_structure_type      (top-level type for the coroutine)
  DW_TAG_variant_part      (variant part)
    DW_AT_discr            (reference to discriminant DW_TAG_member)
    DW_TAG_member          (discriminant member)
    DW_TAG_variant         (variant 1)
    DW_TAG_variant         (variant 2)
    DW_TAG_variant         (variant 3)
  DW_TAG_structure_type    (type of variant 1)
  DW_TAG_structure_type    (type of variant 2)
  DW_TAG_structure_type    (type of variant 3)
```

#### PDB
PDB does not have a dedicated node, so it generates the C equivalent of a discriminated union:

```c
union enum2$<RUST_ENUM_NAME> {
    enum VariantNames {
        First,
        Second
    };
    struct Variant0 {
        struct First {
            // fields
        };
        static const enum2$<RUST_ENUM_NAME>::VariantNames NAME;
        static const unsigned long DISCR_EXACT;
        enum2$<RUST_ENUM_NAME>::Variant0::First value;
    };
    struct Variant1 {
        struct Second {
            // fields
        };
        static enum2$<RUST_ENUM_NAME>::VariantNames NAME;
        static unsigned long DISCR_EXACT;
        enum2$<RUST_ENUM_NAME>::Variant1::Second value;
    };
    enum2$<RUST_ENUM_NAME>::Variant0 variant0;
    enum2$<RUST_ENUM_NAME>::Variant1 variant1;
    unsigned long tag;
}
```

An important note is that due to limitations in LLDB, the `DISCR_*` value generated is always a
`u64` even if the value is not `#[repr(u64)]`. This is largely a non-issue for LLDB because the
`DISCR_*` value and the `tag` are read into `uint64_t` values regardless of their type.

# Source Information

TODO