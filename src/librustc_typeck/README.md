NB: This crate is part of the Rust compiler. For an overview of the
compiler as a whole, see
[the README.md file found in `librustc`](../librustc/README.md).

The `rustc_typeck` crate contains the source for "type collection" and
"type checking", as well as a few other bits of related functionality.
(It draws heavily on the [type inferencing][infer] and
[trait solving][traits] code found in librustc.)

[infer]: ../librustc/infer/README.md
[traits]: ../librustc/traits/README.md

## Type collection

Type "collection" is the process of converting the types found in the
HIR (`hir::Ty`), which represent the syntactic things that the user
wrote, into the **internal representation** used by the compiler
(`Ty<'tcx>`) -- we also do similar conversions for where-clauses and
other bits of the function signature.

To try and get a sense for the difference, consider this function:

```rust
struct Foo { }
fn foo(x: Foo, y: self::Foo) { .. }
//        ^^^     ^^^^^^^^^
```

Those two parameters `x` and `y` each have the same type: but they
will have distinct `hir::Ty` nodes. Those nodes will have different
spans, and of course they encode the path somewhat differently. But
once they are "collected" into `Ty<'tcx>` nodes, they will be
represented by the exact same internal type.

Collection is defined as a bundle of queries (e.g., `type_of`) for
computing information about the various functions, traits, and other
items in the crate being compiled. Note that each of these queries is
concerned with *interprocedural* things -- for example, for a function
definition, collection will figure out the type and signature of the
function, but it will not visit the *body* of the function in any way,
nor examine type annotations on local variables (that's the job of
type *checking*).

For more details, see the `collect` module.

## Type checking

TODO
