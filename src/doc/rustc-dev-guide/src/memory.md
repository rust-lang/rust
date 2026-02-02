# Memory management in rustc

Generally rustc tries to be pretty careful how it manages memory.
The compiler allocates _a lot_ of data structures throughout compilation,
and if we are not careful, it will take a lot of time and space to do so.

One of the main way the compiler manages this is using [arena]s and [interning].

[arena]: https://en.wikipedia.org/wiki/Region-based_memory_management
[interning]: https://en.wikipedia.org/wiki/String_interning

## Arenas and  Interning

Since A LOT of data structures are created during compilation, for performance
reasons, we allocate them from a global memory pool.
Each are allocated once from a long-lived *arena*.
This is called _arena allocation_.
This system reduces allocations/deallocations of memory.
It also allows for easy comparison of types (more on types [here](./ty.md)) for equality:
for each interned type `X`, we implemented [`PartialEq` for X][peqimpl],
so we can just compare pointers.
The [`CtxtInterners`] type contains a bunch of maps of interned types and the arena itself.

[`CtxtInterners`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.CtxtInterners.html#structfield.arena
[peqimpl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Ty.html#implementations

### Example: `ty::TyKind`

Taking the example of [`ty::TyKind`] which represents a type in the compiler (you
can read more [here](./ty.md)).  Each time we want to construct a type, the
compiler doesn’t naively allocate from the buffer.  Instead, we check if that
type was already constructed. If it was, we just get the same pointer we had
before, otherwise we make a fresh pointer. With this schema if we want to know
if two types are the same, all we need to do is compare the pointers which is
efficient. [`ty::TyKind`] should never be constructed on the stack, and it would be unusable
if done so.
You always allocate them from this arena and you always intern them so they are
unique.

At the beginning of the compilation we make a buffer and each time we need to allocate a type we use
some of this memory buffer. If we run out of space we get another one. The lifetime of that buffer
is `'tcx`. Our types are tied to that lifetime, so when compilation finishes all the memory related
to that buffer is freed and our `'tcx` references would be invalid.

In addition to types, there are a number of other arena-allocated data structures that you can
allocate, and which are found in this module. Here are a few examples:

- [`GenericArgs`], allocated with [`mk_args`] – this will intern a slice of types, often used
to specify the values to be substituted for generics args (e.g. `HashMap<i32, u32>` would be
represented as a slice `&'tcx [tcx.types.i32, tcx.types.u32]`).
- [`TraitRef`], typically passed by value – a **trait reference** consists of a reference to a trait
  along with its various type parameters (including `Self`), like `i32: Display` (here, the def-id
  would reference the `Display` trait, and the args would contain `i32`). Note that `def-id` is
  defined and discussed in depth in the [`AdtDef and DefId`][adtdefid] section.
- [`Predicate`] defines something the trait system has to prove (see [traits] module).

[`GenericArgs`]: ./ty_module/generic_arguments.md#the-genericargs-type
[adtdefid]: ./ty_module/generic_arguments.md#adtdef-and-defid
[`TraitRef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TraitRef.html
[`AdtDef` and `DefId`]: ./ty.md#adts-representation
[`def-id`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/def_id/struct.DefId.html
[`GenericArgs`]: ./generic_arguments.html#GenericArgs
[`mk_args`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.mk_args
[adtdefid]: ./ty_module/generic_arguments.md#adtdef-and-defid
[`Predicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.Predicate.html
[`TraitRef`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/type.TraitRef.html
[`ty::TyKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/sty/type.TyKind.html
[traits]: ./traits/resolution.md

## The `tcx` and how it uses lifetimes

The typing context (`tcx`) is the central data structure in the compiler. It is the context that
you use to perform all manner of queries. The `struct` [`TyCtxt`] defines a reference to this shared
context:

```rust,ignore
tcx: TyCtxt<'tcx>
//          ----
//          |
//          arena lifetime
```

As you can see, the `TyCtxt` type takes a lifetime parameter. When you see a reference with a
lifetime like `'tcx`, you know that it refers to arena-allocated data (or data that lives as long as
the arenas, anyhow).

[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html

### A Note On Lifetimes

The Rust compiler is a fairly large program containing lots of big data
structures (e.g. the [Abstract Syntax Tree (AST)][ast], [High-Level Intermediate
Representation (`HIR`)][hir], and the type system) and as such, arenas and
references are heavily relied upon to minimize unnecessary memory use. This
manifests itself in the way people can plug into the compiler (i.e. the
[driver](./rustc-driver/intro.md)), preferring a "push"-style API (callbacks) instead
of the more Rust-ic "pull" style (think the `Iterator` trait).

Thread-local storage and interning are used a lot through the compiler to reduce
duplication while also preventing a lot of the ergonomic issues due to many
pervasive lifetimes. The [`rustc_middle::ty::tls`][tls] module is used to access these
thread-locals, although you should rarely need to touch it.

[ast]: ./ast-validation.md
[hir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/index.html
[tls]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/tls/index.html
