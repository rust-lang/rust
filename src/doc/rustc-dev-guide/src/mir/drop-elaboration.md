# Drop elaboration

## Dynamic drops

According to the [reference][reference-drop]:

> When an initialized variable or temporary goes out of scope, its destructor
> is run, or it is dropped. Assignment also runs the destructor of its
> left-hand operand, if it's initialized. If a variable has been partially
> initialized, only its initialized fields are dropped.

When building the MIR, the `Drop` and `DropAndReplace` terminators represent
places where drops may occur. However, in this phase, the presence of these
terminators does not guarantee that a destructor will run. That's because the
target of a drop may be uninitialized (usually because it has been moved from)
before the terminator is reached. In general, we cannot know at compile-time whether a
variable is initialized.

```rust
let mut y = vec![];

{
    let x = vec![1, 2, 3];
    if std::process::id() % 2 == 0 {
        y = x; // conditionally move `x` into `y`
    }
} // `x` goes out of scope here. Should it be dropped?
```

In these cases, we need to keep track of whether a variable is initialized
*dynamically*. The rules are laid out in detail in [RFC 320: Non-zeroing
dynamic drops][RFC 320].

## Drop obligations

From the RFC:

> When a local variable becomes initialized, it establishes a set of "drop
> obligations": a set of structural paths (e.g. a local `a`, or a path to a
> field `b.f.y`) that need to be dropped.
>
> The drop obligations for a local variable x of struct-type `T` are computed
> from analyzing the structure of `T`. If `T` itself implements `Drop`, then `x` is a
> drop obligation. If `T` does not implement `Drop`, then the set of drop
> obligations is the union of the drop obligations of the fields of `T`.

When a structural path is moved from (and thus becomes uninitialized), any drop
obligations for that path or its descendants (`path.f`, `path.f.g.h`, etc.) are
released. Types with `Drop` implementations do not permit moves from individual
fields, so there is no need to track initializedness through them.

When a local variable goes out of scope (`Drop`), or when a structural path is
overwritten via assignment (`DropAndReplace`), we check for any drop
obligations for that variable or path.  Unless that obligation has been
released by this point, its associated `Drop` implementation will be called.
For `enum` types, only fields corresponding to the "active" variant need to be
dropped. When processing drop obligations for such types, we first check the
discriminant to determine the active variant. All drop obligations for variants
besides the active one are ignored.

Here are a few interesting types to help illustrate these rules:

```rust
struct NoDrop(u8); // No `Drop` impl. No fields with `Drop` impls.

struct NeedsDrop(Vec<u8>); // No `Drop` impl but has fields with `Drop` impls.

struct ThinVec(*const u8); // Custom `Drop` impl. Individual fields cannot be moved from.

impl Drop for ThinVec {
    fn drop(&mut self) { /* ... */ }
}

enum MaybeDrop {
    Yes(NeedsDrop),
    No(NoDrop),
}
```

## Drop elaboration

One valid model for these rules is to keep a boolean flag (a "drop flag") for
every structural path that is used at any point in the function. This flag is
set when its path is initialized and is cleared when the path is moved from.
When a `Drop` occurs, we check the flags for every obligation associated with
the target of the `Drop` and call the associated `Drop` impl for those that are
still applicable.

This process—transforming the newly built MIR with its imprecise `Drop` and
`DropAndReplace` terminators into one with drop flags—is known as drop
elaboration. When a MIR statement causes a variable to become initialized (or
uninitialized), drop elaboration inserts code that sets (or clears) the drop
flag for that variable. It wraps `Drop` terminators in conditionals that check
the newly inserted drop flags.

Drop elaboration also splits `DropAndReplace` terminators into a `Drop` of the
target and a write of the newly dropped place. This is somewhat unrelated to what
we've discussed above.

Once this is complete, `Drop` terminators in the MIR correspond to a call to
the "drop glue" or "drop shim" for the type of the dropped place. The drop
glue for a type calls the `Drop` impl for that type (if one exists), and then
recursively calls the drop glue for all fields of that type.

## Drop elaboration in `rustc`

The approach described above is more expensive than necessary. One can imagine
a few optimizations:

- Only paths that are the target of a `Drop` (or have the target as a prefix)
  need drop flags.
- Some variables are known to be initialized (or uninitialized) when they are
  dropped. These do not need drop flags.
- If a set of paths are only dropped or moved from via a shared prefix, those
  paths can share a single drop flag.

A subset of these are implemented in `rustc`.

In the compiler, drop elaboration is split across several modules. The pass
itself is defined [here][drops-transform], but the [main logic][drops] is
defined elsewhere since it is also used to build [drop shims][drops-shim].

Drop elaboration designates each `Drop` in the newly built MIR as one of four
kinds:

- `Static`, the target is always initialized.
- `Dead`, the target is always **un**initialized.
- `Conditional`, the target is either wholly initialized or wholly
  uninitialized. It is not partly initialized.
- `Open`, the target may be partly initialized.

For this, it uses a pair of dataflow analyses, `MaybeInitializedPlaces` and
`MaybeUninitializedPlaces`. If a place is in one but not the other, then the
initializedness of the target is known at compile-time (`Dead` or `Static`).
In this case, drop elaboration does not add a flag for the target. It simply
removes (`Dead`) or preserves (`Static`) the `Drop` terminator.

For `Conditional` drops, we know that the initializedness of the variable as a
whole is the same as the initializedness of its fields. Therefore, once we
generate a drop flag for the target of that drop, it's safe to call the drop
glue for that target.

### `Open` drops

`Open` drops are the most complex, since we need to break down a single `Drop`
terminator into several different ones, one for each field of the target whose
type has drop glue (`Ty::needs_drop`). We cannot call the drop glue for the
target itself because that requires all fields of the target to be initialized.
Remember, variables whose type has a custom `Drop` impl do not allow `Open`
drops because their fields cannot be moved from.

This is accomplished by recursively categorizing each field as `Dead`,
`Static`, `Conditional` or `Open`. Fields whose type does not have drop glue
are automatically `Dead` and need not be considered during the recursion. When
we reach a field whose kind is not `Open`, we handle it as we did above. If the
field is also `Open`, the recursion continues.

It's worth noting how we handle `Open` drops of enums. Inside drop elaboration,
each variant of the enum is treated like a field, with the invariant that only
one of those "variant fields" can be initialized at any given time. In the
general case, we do not know which variant is the active one, so we will have
to call the drop glue for the enum (which checks the discriminant) or check the
discriminant ourselves as part of an elaborated `Open` drop. However, in
certain cases (within a `match` arm, for example) we do know which variant of
an enum is active. This information is encoded in the `MaybeInitializedPlaces`
and `MaybeUninitializedPlaces` dataflow analyses by marking all places
corresponding to inactive variants as uninitialized.

### Cleanup paths

TODO: Discuss drop elaboration and unwinding.

## Aside: drop elaboration and const-eval

In Rust, functions that are eligible for evaluation at compile-time must be
marked explicitly using the `const` keyword. This includes implementations  of
the `Drop` trait, which may or may not be `const`. Code that is eligible for
compile-time evaluation may only call `const` functions, so any calls to
non-const `Drop` implementations in such code must be forbidden.

A call to a `Drop` impl is encoded as a `Drop` terminator in the MIR. However,
as we discussed above, a `Drop` terminator in newly built MIR does not
necessarily result in a call to `Drop::drop`. The drop target may be
uninitialized at that point. This means that checking for non-const `Drop`s on
the newly built MIR can result in spurious errors. Instead, we wait until after
drop elaboration runs, which eliminates `Dead` drops (ones where the target is
known to be uninitialized) to run these checks.

[RFC 320]: https://rust-lang.github.io/rfcs/0320-nonzeroing-dynamic-drop.html
[reference-drop]: https://doc.rust-lang.org/reference/destructors.html
[drops]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_mir_dataflow/src/elaborate_drops.rs
[drops-shim]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_mir_transform/src/shim.rs
[drops-transform]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_mir_transform/src/elaborate_drops.rs
