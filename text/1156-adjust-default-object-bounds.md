- Feature Name: N/A
- Start Date: 2015-06-4
- RFC PR: https://github.com/rust-lang/rfcs/pull/1156
- Rust Issue: https://github.com/rust-lang/rust/issues/26438

# Summary

Adjust the object default bound algorithm for cases like `&'x
Box<Trait>` and `&'x Arc<Trait>`. The existing algorithm would default
to `&'x Box<Trait+'x>`. The proposed change is to default to `&'x
Box<Trait+'static>`.

Note: This is a **BREAKING CHANGE**. The change has
[been implemented][branch] and its impact has been evaluated. It was
[found][crater] to cause **no root regressions** on `crates.io`.
Nonetheless, to minimize impact, this RFC proposes phasing in the
change as follows:

- In Rust 1.2, a warning will be issued for code which will break when the
  defaults are changed. This warning can be disabled by using explicit
  bounds. The warning will only be issued when explicit bounds would be required
  in the future anyway.
- In Rust 1.3, the change will be made permanent. Any code that has
  not been updated by that time will break.

# Motivation

When we instituted default object bounds, [RFC 599] specified that
`&'x Box<Trait>` (and `&'x mut Box<Trait>`) should expand to `&'x
Box<Trait+'x>` (and `&'x mut Box<Trait+'x>`). This is in contrast to a
`Box` type that appears outside of a reference (e.g., `Box<Trait>`),
which defaults to using `'static` (`Box<Trait+'static>`). This
decision was made because it meant that a function written like so
would accept the broadest set of possible objects:

```rust
fn foo(x: &Box<Trait>) {
}
```

In particular, under the current defaults, `foo` can be supplied an
object which references borrowed data. Given that `foo` is taking the
argument by reference, it seemed like a good rule. Experience has
shown otherwise (see below for some of the problems encountered).

This RFC proposes changing the default object bound rules so that the
default is drawn from the innermost type that encloses the trait
object. If there is no such type, the default is `'static`. The type
is a reference (e.g., `&'r Trait`), then the default is the lifetime
`'r` of that reference. Otherwise, the type must in practice be some
user-declared type, and the default is derived from the declaration:
if the type declares a lifetime bound, then this lifetime bound is
used, otherwise `'static` is used. This means that (e.g.) `&'r
Box<Trait>` would default to `&'r Box<Trait+'static>`, and `&'r
Ref<'q, Trait>` (from `RefCell`) would default to `&'r Ref<'q,
Trait+'q>`.

### Problems with the current default.

**Same types, different expansions.** One problem is fairly
predictable: the current default means that identical types differ in
their interpretation based on where they appear. This is something we
have striven to avoid in general. So, as an example, this code
[will not type-check](http://is.gd/Yaak1l):

```rust
trait Trait { }

struct Foo {
    field: Box<Trait>
}

fn do_something(f: &mut Foo, x: &mut Box<Trait>) {
    mem::swap(&mut f.field, &mut *x);
}
```

Even though `x` is a reference to a `Box<Trait>` and the type of
`field` is a `Box<Trait>`, the expansions differ. `x` expands to `&'x
mut Box<Trait+'x>` and the field expands to `Box<Trait+'static>`.  In
general, we have tried to ensure that if the type is *typed precisely
the same* in a type definition and a fn definition, then those two
types are equal (note that fn definitions allow you to omit things
that cannot be omitted in types, so some types that you can enter in a
fn definition, like `&i32`, cannot appear in a type definition).

Now, the same is of course true for the type `Trait` itself, which
appears identically in different contexts and is expanded in different
ways. This is not a problem here because the type `Trait` is unsized,
which means that it cannot be swapped or moved, and hence the main
sources of type mismatches are avoided.

**Mental model.** In general the mental model of the newer rules seems
simpler: once you move a trait object into the heap (via `Box`, or
`Arc`), you must explicitly indicate whether it can contain borrowed
data or not.  So long as you manipulate by reference, you don't have
to. In contrast, the current rules are more subtle, since objects in
the heap may still accept borrowed data, if you have a reference to
the box.

**Poor interaction with the dropck rules.** When implementing the
newer dropck rules specified by [RFC 769], we found a
[rather subtle problem] that would arise with the current defaults.
The precise problem is spelled out in appendix below, but the TL;DR is
that if you wish to pass an array of boxed objects, the current
defaults can be actively harmful, and hence force you to specify
explicit lifetimes, whereas the newer defaults do something
reasonable.

# Detailed design

The rules for user-defined types from RFC 599 are altered as follows
(text that is not changed is italicized):

- *If `SomeType` contains a single where-clause like `T:'a`, where
  `T` is some type parameter on `SomeType` and `'a` is some
  lifetime, then the type provided as value of `T` will have a
  default object bound of `'a`. An example of this is
  `std::cell::Ref`: a usage like `Ref<'x, X>` would change the
  default for object types appearing in `X` to be `'a`.*
- If `SomeType` contains no where-clauses of the form `T:'a`, then
  the "base default" is used. The base default depends on the overall context:
  - in a fn body, the base default is a fresh inference variable.
  - outside of a fn body, such in a fn signature, the base default
    is `'static`.
  Hence `Box<X>` would typically be a default of `'static` for `X`,
  regardless of whether it appears underneath an `&` or not.
  (Note that in a fn body, the inference is strong enough to adopt `'static`
  if that is the necessary bound, or a looser bound if that would be helpful.)
- *If `SomeType` contains multiple where-clauses of the form `T:'a`,
  then the default is cleared and explicit lifetiem bounds are
  required. There are no known examples of this in the standard
  library as this situation arises rarely in practice.*

# Timing and breaking change implications

This is a breaking change, and hence it behooves us to evaluate the
impact and describe a procedure for making the change as painless as
possible. One nice propery of this change is that it only affects
*defaults*, which means that it is always possible to write code that
compiles both before and after the change by avoiding defaults in
those cases where the new and old compiler disagree.

The estimated impact of this change is very low, for two reasons:
- A recent test of crates.io found [no regressions][crater] caused by
  this change (however, a [previous run] (from before Rust 1.0) found 8
  regressions).
- This feature was only recently stabilized as part of Rust 1.0 (and
  was only added towards the end of the release cycle), so there
  hasn't been time for a large body of dependent code to arise
  outside of crates.io.

Nonetheless, to minimize impact, this RFC proposes phasing in the
change as follows:

- In Rust 1.2, a warning will be issued for code which will break when the
  defaults are changed. This warning can be disabled by using explicit
  bounds. The warning will only be issued when explicit bounds would be required
  in the future anyway.
  - Specifically, types that were written `&Box<Trait>` where the
    (boxed) trait object may contain references should now be written
    `&Box<Trait+'a>` to disable the warning.
- In Rust 1.3, the change will be made permanent. Any code that has
  not been updated by that time will break.

# Drawbacks

The primary drawback is that this is a breaking change, as discussed
in the previous section.

# Alternatives

Keep the current design, with its known drawbacks.

# Unresolved questions

None.

# Appendix: Details of the dropck problem

This appendix goes into detail about the sticky interaction with
dropck that was uncovered. The problem arises if you have a function
that wishes to take a mutable slice of objects, like so:

```rust
fn do_it(x: &mut [Box<FnMut()>]) { ... }
```

Here, `&mut [..]` is used because the objects are `FnMut` objects, and
hence require `&mut self` to call. This function in turn is expanded
to:

```rust
fn do_it<'x>(x: &'x mut [Box<FnMut()+'x>]) { ... }
```

Now callers might try to invoke the function as so:

```rust
do_it(&mut [Box::new(val1), Box::new(val2)])
```

Unfortunately, this code fails to compile -- in fact, it cannot be
made to compile without changing the definition of `do_it`, due to a
sticky interaction between dropck and variance. The problem is that
dropck requires that all data in the box strictly outlives the
lifetime of the box's owner. This is to prevent cyclic
content. Therefore, the type of the objects must be `Box<FnMut()+'R>`
where `'R` is some region that strictly outlives the array itself (as
the array is the owner of the objects).  However, the signature of
`do_it` demands that the reference to the array has the same lifetime
as the trait objects within (and because this is an `&mut` reference
and hence invariant, no approximation is permitted). This implies that
the array must live for at least the region `'R`. But we defined the
region `'R` to be some region that outlives the array, so we have a
quandry.

The solution is to change the definition of `do_it` in one of two
ways:

```rust
// Use explicit lifetimes to make it clear that the reference is not
// required to have the same lifetime as the objects themselves:
fn do_it1<'a,'b>(x: &'a mut [Box<FnMut()+'b>]) { ... }

// Specifying 'static is easier, but then the closures cannot
// capture the stack:
fn do_it2(x: &'a mut [Box<FnMut()+'static>]) { ... }
```

Under the proposed RFC, `do_it2` would be the default.  If one wanted
to use lifetimes, then one would have to use explicit lifetime
overrides as shown in `do_it1`. This is consistent with the mental
model of "once you box up an object, you must add annotations for it
to contain borrowed data".

[RFC 599]: 0599-default-object-bound.md
[RFC 769]: 0769-sound-generic-drop.md
[rather subtle problem]: https://github.com/rust-lang/rust/pull/25212#issuecomment-100244929
[crater]: https://gist.github.com/brson/085d84d43c6a9a8d4dc3
[branch]: https://github.com/nikomatsakis/rust/tree/better-object-defaults
[previous run]: https://gist.github.com/brson/80f9b80acef2e7ab37ee
[RFC 1122]: https://github.com/rust-lang/rfcs/pull/1122
