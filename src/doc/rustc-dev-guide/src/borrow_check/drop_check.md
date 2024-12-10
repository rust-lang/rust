# Drop Check

We generally require the type of locals to be well-formed whenever the
local is used. This includes proving the where-bounds of the local and
also requires all regions used by it to be live.

The only exception to this is when implicitly dropping values when they
go out of scope. This does not necessarily require the value to be live:

```rust
fn main() {
    let x = vec![];
    {
        let y = String::from("I am temporary");
        x.push(&y);
    }
    // `x` goes out of scope here, after the reference to `y`
    // is invalidated. This means that while dropping `x` its type
    // is not well-formed as it contain regions which are not live.
}
```

This is only sound if dropping the value does not try to access any dead
region. We check this by requiring the type of the value to be
drop-live.
The requirements for which are computed in `fn dropck_outlives`.

The rest of this section uses the following type definition for a type
which requires its region parameter to be live:

```rust
struct PrintOnDrop<'a>(&'a str);
impl<'a> Drop for PrintOnDrop<'_> {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}
```

## How values are dropped

At its core, a value of type `T` is dropped by executing its "drop
glue". Drop glue is compiler generated and first calls `<T as
Drop>::drop` and then recursively calls the drop glue of any recursively
owned values.

- If `T` has an explicit `Drop` impl, call `<T as Drop>::drop`.
- Regardless of whether `T` implements `Drop`, recurse into all values
  *owned* by `T`:
    - references, raw pointers, function pointers, function items, trait
      objects[^traitobj], and scalars do not own anything.
    - tuples, slices, and arrays consider their elements to be owned.
      For arrays of length zero we do not own any value of the element
      type.
    - all fields (of all variants) of ADTs are considered owned. We
      consider all variants for enums. The exception here is
      `ManuallyDrop<U>` which is not considered to own `U`.
      `PhantomData<U>` also does not own anything.
      closures and generators own their captured upvars.

Whether a type has drop glue is returned by [`fn
Ty::needs_drop`](https://github.com/rust-lang/rust/blob/320b412f9c55bf480d26276ff0ab480e4ecb29c0/compiler/rustc_middle/src/ty/util.rs#L1086-L1108).

### Partially dropping a local

For types which do not implement `Drop` themselves, we can also
partially move parts of the value before dropping the rest. In this case
only the drop glue for the not-yet moved values is called, e.g.

```rust
fn main() {
    let mut x = (PrintOnDrop("third"), PrintOnDrop("first"));
    drop(x.1);
    println!("second")
}
```

During MIR building we assume that a local may get dropped whenever it
goes out of scope *as long as its type needs drop*. Computing the exact
drop glue for a variable happens **after** borrowck in the
`ElaborateDrops` pass. This means that even if some part of the local
have been dropped previously, dropck still requires this value to be
live. This is the case even if we completely moved a local.

```rust
fn main() {
    let mut x;
    {
        let temp = String::from("I am temporary");
        x = PrintOnDrop(&temp);
        drop(x);
    }
} //~ ERROR `temp` does not live long enough.
```

It should be possible to add some amount of drop elaboration before
borrowck, allowing this example to compile. There is an unstable feature
to move drop elaboration before const checking:
[#73255](https://github.com/rust-lang/rust/issues/73255). Such a feature
gate does not exist for doing some drop elaboration before borrowck,
although there's a [relevant
MCP](https://github.com/rust-lang/compiler-team/issues/558).

[^traitobj]: you can consider trait objects to have a builtin `Drop`
implementation which directly uses the `drop_in_place` provided by the
vtable. This `Drop` implementation requires all its generic parameters
to be live.

### `dropck_outlives`

There are two distinct "liveness" computations that we perform:

* a value `v` is *use-live* at location `L` if it may be "used" later; a
  *use* here is basically anything that is not a *drop*
* a value `v` is *drop-live* at location `L` if it maybe dropped later

When things are *use-live*, their entire type must be valid at `L`. When
they are *drop-live*, all regions that are required by dropck must be
valid at `L`.  The values dropped in the MIR are *places*.

The constraints computed by `dropck_outlives` for a type closely match
the generated drop glue for that type. Unlike drop glue,
`dropck_outlives` cares about the types of owned values, not the values
itself. For a value of type `T`

- if `T` has an explicit `Drop`, require all generic arguments to be
  live, unless they are marked with `#[may_dangle]` in which case they
  are fully ignored
- regardless of whether `T` has an explicit `Drop`, recurse into all
  types *owned* by `T`
    - references, raw pointers, function pointers, function items, trait
      objects[^traitobj], and scalars do not own anything.
    - tuples, slices and arrays consider their element type to be owned.
      **For arrays we currently do not check whether their length is
      zero**.
    - all fields (of all variants) of ADTs are considered owned. The
      exception here is `ManuallyDrop<U>` which is not considered to own
      `U`. **We consider `PhantomData<U>` to own `U`**.
    - closures and generators own their captured upvars.

The sections marked in bold are cases where `dropck_outlives` considers
types to be owned which are ignored by `Ty::needs_drop`. We only rely on
`dropck_outlives` if `Ty::needs_drop` for the containing local returned
`true`.This means liveness requirements can change depending on whether
a type is contained in a larger local. **This is inconsistent, and
should be fixed: an example [for
arrays](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=8b5f5f005a03971b22edb1c20c5e6cbe)
and [for
`PhantomData`](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=44c6e2b1fae826329fd54c347603b6c8).**[^core]

One possible way these inconsistencies can be fixed is by MIR building
to be more pessimistic, probably by making `Ty::needs_drop` weaker, or
alternatively, changing `dropck_outlives` to be more precise, requiring
fewer regions to be live.

[^core]: This is the core assumption of [#110288](https://github.com/rust-lang/rust/issues/110288) and [RFC 3417](https://github.com/rust-lang/rfcs/pull/3417).
