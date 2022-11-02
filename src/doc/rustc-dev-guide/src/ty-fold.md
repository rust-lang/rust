# `TypeFoldable` and `TypeFolder`

How is this `subst` query actually implemented? As you can imagine, we might want to do
substitutions on a lot of different things. For example, we might want to do a substitution directly
on a type like we did with `Vec` above. But we might also have a more complex type with other types
nested inside that also need substitutions.

The answer is a couple of traits:
[`TypeFoldable`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/fold/trait.TypeFoldable.html)
and
[`TypeFolder`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/fold/trait.TypeFolder.html).

- `TypeFoldable` is implemented by types that embed type information. It allows you to recursively
  process the contents of the `TypeFoldable` and do stuff to them.
- `TypeFolder` defines what you want to do with the types you encounter while processing the
  `TypeFoldable`.

For example, the `TypeFolder` trait has a method
[`fold_ty`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/fold/trait.TypeFolder.html#method.fold_ty)
that takes a type as input and returns a new type as a result. `TypeFoldable` invokes the
`TypeFolder` `fold_foo` methods on itself, giving the `TypeFolder` access to its contents (the
types, regions, etc that are contained within).

You can think of it with this analogy to the iterator combinators we have come to love in rust:

```rust,ignore
vec.iter().map(|e1| foo(e2)).collect()
//             ^^^^^^^^^^^^ analogous to `TypeFolder`
//         ^^^ analogous to `TypeFoldable`
```

So to reiterate:

- `TypeFolder`  is a trait that defines a “map” operation.
- `TypeFoldable`  is a trait that is implemented by things that embed types.

In the case of `subst`, we can see that it is implemented as a `TypeFolder`:
[`SubstFolder`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/subst/struct.SubstFolder.html).
Looking at its implementation, we see where the actual substitutions are happening.

However, you might also notice that the implementation calls this `super_fold_with` method. What is
that? It is a method of `TypeFoldable`. Consider the following `TypeFoldable` type `MyFoldable`:

```rust,ignore
struct MyFoldable<'tcx> {
  def_id: DefId,
  ty: Ty<'tcx>,
}
```

The `TypeFolder` can call `super_fold_with` on `MyFoldable` if it just wants to replace some of the
fields of `MyFoldable` with new values. If it instead wants to replace the whole `MyFoldable` with a
different one, it would call `fold_with` instead (a different method on `TypeFoldable`).

In almost all cases, we don’t want to replace the whole struct; we only want to replace `ty::Ty`s in
the struct, so usually we call `super_fold_with`. A typical implementation that `MyFoldable` could
have might do something like this:

```rust,ignore
my_foldable: MyFoldable<'tcx>
my_foldable.subst(..., subst)

impl TypeFoldable for MyFoldable {
  fn super_fold_with(&self, folder: &mut impl TypeFolder<'tcx>) -> MyFoldable {
    MyFoldable {
      def_id: self.def_id.fold_with(folder),
      ty: self.ty.fold_with(folder),
    }
  }

  fn super_visit_with(..) { }
}
```

Notice that here, we implement `super_fold_with` to go over the fields of `MyFoldable` and call
`fold_with` on *them*. That is, a folder may replace  `def_id` and `ty`, but not the whole
`MyFoldable` struct.

Here is another example to put things together: suppose we have a type like `Vec<Vec<X>>`. The
`ty::Ty` would look like: `Adt(Vec, &[Adt(Vec, &[Param(X)])])`. If we want to do `subst(X => u32)`,
then we would first look at the overall type. We would see that there are no substitutions to be
made at the outer level, so we would descend one level and look at `Adt(Vec, &[Param(X)])`. There
are still no substitutions to be made here, so we would descend again. Now we are looking at
`Param(X)`, which can be substituted, so we replace it with `u32`. We can’t descend any more, so we
are done, and  the overall result is `Adt(Vec, &[Adt(Vec, &[u32])])`.

One last thing to mention: often when folding over a `TypeFoldable`, we don’t want to change most
things. We only want to do something when we reach a type. That means there may be a lot of
`TypeFoldable` types whose implementations basically just forward to their fields’ `TypeFoldable`
implementations. Such implementations of `TypeFoldable` tend to be pretty tedious to write by hand.
For this reason, there is a `derive` macro that allows you to `#![derive(TypeFoldable)]`. It is
defined
[here](https://github.com/rust-lang/rust/blob/master/compiler/rustc_macros/src/type_foldable.rs).

**`subst`** In the case of substitutions the [actual
folder](https://github.com/rust-lang/rust/blob/75ff3110ac6d8a0259023b83fd20d7ab295f8dd6/src/librustc_middle/ty/subst.rs#L440-L451)
is going to be doing the indexing we’ve already mentioned. There we define a `Folder` and call
`fold_with` on the `TypeFoldable` to process yourself.  Then
[fold_ty](https://github.com/rust-lang/rust/blob/75ff3110ac6d8a0259023b83fd20d7ab295f8dd6/src/librustc_middle/ty/subst.rs#L512-L536)
the method that process each type it looks for a `ty::Param` and for those it replaces it for
something from the list of substitutions, otherwise recursively process the type.  To replace it,
calls
[ty_for_param](https://github.com/rust-lang/rust/blob/75ff3110ac6d8a0259023b83fd20d7ab295f8dd6/src/librustc_middle/ty/subst.rs#L552-L587)
and all that does is index into the list of substitutions with the index of the `Param`.

