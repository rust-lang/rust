- Feature Name: pattern-binding-modes
- Start Date: 2016-08-12
- RFC PR: https://github.com/rust-lang/rfcs/pull/2005
- Rust Issue: https://github.com/rust-lang/rust/issues/42640

# Summary
[summary]: #summary

Better ergonomics for pattern-matching on references.

Currently, matching on references requires a bit of a dance using
`ref` and `&` patterns:

```
let x: &Option<_> = &Some(0);

match x {
    &Some(ref y) => { ... },
    &None => { ... },
}

// or using `*`:

match *x {
    Some(ref x) => { ... },
    None => { ... },
}
```

After this RFC, the above form still works, but now we also allow a simpler form:

```
let x: &Option<_> = &Some(0);

match x {
    Some(y) => { ... }, // `y` is a reference to `0`
    None => { ... },
}
```

This is accomplished through automatic dereferencing and the introduction of
default binding modes.

# Motivation
[motivation]: #motivation

Rust is usually strict when distinguishing between value and reference types. In
particular, distinguishing borrowed and owned data. However, there is often a
trade-off between [explicit-ness and ergonomics](https://blog.rust-lang.org/2017/03/02/lang-ergonomics.html),
and Rust errs on the side of ergonomics in some carefully selected places.
Notably when using the dot operator to call methods and access fields, and when
declaring closures.

The match expression is an extremely common expression and arguably, the most
important control flow mechanism in Rust. Borrowed data is probably the most
common form in the language. However, using match expressions and borrowed data
together can be frustrating: getting the correct combination of `*`, `&`, and
`ref` to satisfy the type and borrow checkers is a common problem, and one which
is often encountered early by Rust beginners. It is especially frustrating since
it seems that the compiler can guess what is needed but gives you error messages
instead of helping.

For example, consider the following program:

```
enum E { Foo(...), Bar }

fn f(e: &E) {
    match e { ... }
}

```

It is clear what we want to do here - we want to check which variant `e` is a
reference to. Annoyingly, we have two valid choices:

```
match e {
    &E::Foo(...) => { ... }
    &E::Bar => { ... }
}
```

and

```
match *e {
    E::Foo(...) => { ... }
    E::Bar => { ... }
}
```

The former is more obvious, but requires more noisey syntax (an `&` on every
arm). The latter can appear a bit magical to newcomers - the type checker treats
`*e` as a value, but the borrow checker treats the data as borrowed for the
duration of the match. It also does not work with nested types, `match (*e,)
...` for example is not allowed.

In either case if we further bind variables, we must ensure that we do not
attempt to move data, e.g.,

```
match *e {
    E::Foo(x) => { ... }
    E::Bar => { ... }
}
```

If the type of `x` does not have the `Copy` bound, then this will give a borrow
check error. We must use the `ref` keyword to take a reference: `E::Foo(ref x)`
(or `&E::Foo(ref x)` if we match `e` rather than `*e`).

The `ref` keyword is a pain for Rust beginners, and a bit of a wart for everyone
else. It violates the rule of patterns matching declarations, it is not found
anywhere outside of patterns, and it is often confused with `&`. (See for
example, https://github.com/rust-lang/rust-by-example/issues/390).

Match expressions are an area where programmers often end up playing 'type
Tetris': adding operators until the compiler stops complaining, without
understanding the underlying issues. This serves little benefit - we can make
match expressions much more ergonomic without sacrificing safety or readability.

Match ergonomics has been highlighted as an area for improvement in 2017:
[internals thread](https://internals.rust-lang.org/t/roadmap-2017-productivity-learning-curve-and-expressiveness/4097)
and [Rustconf keynote](https://www.youtube.com/watch?v=pTQxHIzGqFI&list=PLE7tQUdRKcybLShxegjn0xyTTDJeYwEkI&index=1).


# Detailed design
[design]: #detailed-design

This RFC is a refinement of
[the match ergonomics RFC](https://github.com/rust-lang/rfcs/pull/1944). Rather
than using auto-deref and auto-referencing, this RFC introduces _default binding
modes_ used when a reference value is matched by a non-reference pattern.

In other words, we allow auto-dereferencing values during pattern-matching.
When an auto-dereference occurs, the compiler will automatically treat the inner
bindings as `ref` or `ref mut` bindings.

Example:

```rust
let x = Some(3);
let y: &Option<i32> = &x;
match y {
  Some(a) => {
    // `y` is dereferenced, and `a` is bound like `ref a`.
  }
  None => {}
}
```

Note that this RFC applies to all instances of pattern-matching, not just
`match` expressions:

```rust
struct Foo(i32);

let foo = Foo(6);
let foo_ref = &foo;
// `foo_ref` is dereferenced, and `x` is bound like `ref x`.
let Foo(x) = foo_ref;
```


## Definitions

A reference pattern is any pattern which can match a reference without
coercion. Reference patterns include bindings, wildcards (`_`),
`const`s of reference types, and patterns beginning with `&` or `&mut`. All
other patterns are _non-reference patterns_.

_Default binding mode_: this mode, either `move`, `ref`, or `ref mut`, is used
to determine how to bind new pattern variables.
When the compiler sees a variable binding not explicitly marked
`ref`, `ref mut`, or `mut`, it uses the _default binding mode_
to determine how the variable should be bound.
Currently, the _default binding mode_ is always `move`.
Under this RFC, matching a reference with a _non-reference pattern_, would shift
the default binding mode to `ref` or `ref mut`.

## Binding mode rules

The _default binding mode_ starts out as `move`. When matching a pattern, the
compiler starts from the outside of the pattern and works inwards.
Each time a reference is matched using a _non-reference pattern_,
it will automatically dereference the value and update the default binding mode:

1. If the reference encountered is `&val`, set the default binding mode to `ref`.
2. If the reference encountered is `&mut val`: if the current default
binding mode is `ref`, it should remain `ref`. Otherwise, set the current binding
mode to `ref mut`.

If the automatically dereferenced value is still a reference, it is dereferenced
and this process repeats.

```
                        Start                                
                          |                                  
                          v                                  
                +-----------------------+                     
                | Default Binding Mode: |                     
                |        move           |                     
                +-----------------------+                     
               /                        \                     
Encountered   /                          \  Encountered       
  &mut val   /                            \     &val
            v                              v                  
+-----------------------+        +-----------------------+    
| Default Binding Mode: |        | Default Binding Mode: |    
|        ref mut        |        |        ref            |    
+-----------------------+        +-----------------------+    
                          ----->                              
                        Encountered                           
                            &val
```

Note that there is no exit from the `ref` binding mode. This is because an
`&mut` inside of a `&` is still a shared reference, and thus cannot be used to
mutate the underlying value.

Also note that no transitions are taken when using an explicit `ref` or
`ref mut` binding. The _default binding mode_ only changes when matching a
reference with a non-reference pattern.

The above rules and the examples that follow are drawn from @nikomatsakis's
[comment proposing this design](https://github.com/rust-lang/rfcs/pull/1944#issuecomment-296133645).

## Examples

No new behavior:
```rust
match &Some(3) {
    p => {
        // `p` is a variable binding. Hence, this is **not** a ref-defaulting
        // match, and `p` is bound with `move` semantics
        // (and has type `&Option<i32>`).
    },
}
```

One match arm with new behavior:
```rust
match &Some(3) {
    Some(p) => {
        // This pattern is not a `const` reference, `_`, or `&`-pattern,
        // so this is a "non-reference pattern."
        // We dereference the `&` and shift the
        // default binding mode to `ref`. `p` is read as `ref p` and given
        // type `&i32`.
    },
    x => {
        // In this arm, we are still in `move`-mode by default, so `x` has type
        // `&Option<i32>`
    },
}

// Desugared:
match &Some(3) {
  &Some(ref P) => {
    ...
  },
  x => {
    ...
  },
}
```

`match` with "or" (`|`) patterns:
```rust
let x = &Some((3, 3));
match x {
  // Here, each of the patterns are treated independently
  Some((x, 3)) | &Some((ref x, 5)) => { ... }
  _ => { ... }
}

// Desugared:
let x = &Some(3);
match x {
  &Some((ref x, 3)) | &Some((ref x, 5)) => { ... }
  None => { ... }
}
```

Multiple nested patterns with new and old behavior, respectively:
```rust
match (&Some(5), &Some(6)) {
    (Some(a), &Some(mut b)) => {
        // Here, the `a` will be `&i32`, because in the first half of the tuple
        // we hit a non-reference pattern and shift into `ref` mode.
        //
        // In the second half of the tuple there's no non-reference pattern,
        // so `b` will be `i32` (bound with `move` mode). Moreover, `b` is
        // mutable.
    },
    _ => { ... }
}

// Desugared:
match (&Some(5), &Some(6)) {
  (&Some(ref a), &Some(mut b)) => {
    ...
  },
  _  => { ... },
}
```

Example with multiple dereferences:
```rust
let x = (1, &Some(5));
let y = &Some(x);
match y {
  Some((a, Some(b))) => { ... }
  _ => { ... }
}

// Desugared:
let x = (1, &Some(5));
let y = &Some(x);
match y {
  &Some((ref a, &Some(ref b))) => { ... }
  _ => { ... }
}
```

Example with nested references:
```rust
let x = &Some(5);
let y = &x;
match y {
    Some(z) => { ... }
    _ => { ... }
}

// Desugared:
let x = &Some(5);
let y = &x;
match y {
    &&Some(ref z) => { ... }
    _ => { ... }
}
```

Example of new mutable reference behavior:
```rust
match &mut x {
    Some(y) => {
        // `y` is an `&mut` reference here, equivalent to `ref mut` before
    },
    None => { ... },
}

// Desugared:
match &mut x {
  &mut Some(ref mut y) => {
    ...
  },
  &mut None => { ... },
}
```

Example using `let`:
```rust
struct Foo(i32);

// Note that these rules apply to any pattern matching
// whether it be in a `match` or a `let`.
// For example, `x` here is a `ref` binding:
let Foo(x) = &Foo(3);

// Desugared:
let &Foo(ref x) = &Foo(3);
```


## Backwards compatibility

In order to guarantee backwards-compatibility, this proposal only modifies
pattern-matching when a reference is matched with a non-reference pattern,
which is an error today.

This reasoning requires that the compiler knows if the type being matched is a
reference, which isn't always true for inference variables.
If the type being matched may
or may not be a reference _and_ it is being matched by a _non-reference
pattern_, then the compiler will default to assuming that it is not a
reference, in which case the binding mode will default to `move` and it will
behave exactly as it does today.

Example:

```rust
let x = vec![];

match x[0] { // This will panic, but that doesn't matter for this example

    // When matching here, we don't know whether `x[0]` is `Option<_>` or
    // `&Option<_>`. `Some(y)` is a non-reference pattern, so we assume that
    // `x[0]` is not a reference
    Some(y) => {

        // Since we know `Vec::contains` takes `&T`, `x` must be of type
        // `Vec<Option<usize>>`. However, we couldn't have known that before
        // analyzing the match body.
        if x.contains(&Some(5)) {
            ...
        }
    }
    None => {}
}
```

# How We Teach This
[how_we_teach_this]: #how_we_teach_this

This RFC makes matching on references easier and less error-prone. The
documentation for matching references should be updated to use the style
outlined in this RFC. Eventually, documentation and error messages should be
updated to phase-out `ref` and `ref mut` in favor of the new, simpler syntax.

# Drawbacks
[drawbacks]: #drawbacks

The major downside of this proposal is that it complicates the pattern-matching
logic. However, doing so allows common cases to "just work", making the beginner
experience more straightforward and requiring fewer manual reference gymnastics.

# Future Extensions
[future extensions]: #future_extensions
In the future, this RFC could be extended to add support for autodereferencing
custom smart-pointer types using the `Deref` and `DerefMut` traits.

```rust
let x: Box<Option<i32>> = Box::new(Some(0));
match &x {
    Some(y) => { ... }, // y: &i32
    None => { ... },
}
```

This feature has been omitted from this RFC. A few of the details of this
feature are unclear, especially when considering interactions with a
future `DerefMove` trait or similar.

Nevertheless, a followup RFC should be able to backwards-compatibly add support
for custom autodereferencable types.

# Alternatives
[alternatives]: #alternatives

1. We could only infer `ref`, leaving users to manually specify the `mut` in
`ref mut` bindings. This has the advantage of keeping mutability explicit.
Unfortunately, it also has some unintuitive results. `ref mut` doesn't actually
produce mutable bindings-- it produces immutably-bound mutable references.
```rust
// Today's behavior:
let mut x = Some(5);
let mut z = 6;
if let Some(ref mut y) = *(&mut x) {
    // `y` here is actually an immutable binding.
    // `y` can be used to mutate the value of `x`, but `y` can't be rebound to
    // a new reference.
    y = &mut z; //~ ERROR: re-assignment of immutable variable `y`
}

// With this RFC's behavior:
let mut x = Some(5);
let mut z = 6;
if let Some(y) = &mut x {
    // The error is the same as above-- `y` is an immutable binding.
    y = &mut z; //~ ERROR: re-assignment of immutable variable `y`
}

// If we modified this RFC to require explicit `mut` annotations:
let mut x = Some(5);
let mut z = 6;
if let Some(mut y) = &mut x {
    // The error is the same, but is now horribly confusing.
    // `y` is clearly labeled `mut`, but it can't be modified.
    y = &mut z; //~ ERROR: re-assignment of immutable variable `y`
}
```
Additionally, we don't require `mut` when declaring immutable reference bindings
today:
```rust
// Today's behavior:
let mut x = Some(5);
// `y` here isn't declared as `mut`, even though it can be used to mutate `x`.
let y = &mut x;
*y = None;
```
Forcing users to manually specify `mut` in reference bindings would
be inconsistent with Rust's current semantics, and would result in confusing
errors.

2. We could support auto-ref / deref as suggested in
[the original match ergonomics RFC.](https://github.com/rust-lang/rfcs/pull/1944)
This approach has troublesome interaction with
backwards-compatibility, and it becomes more difficult for the user to reason
about whether they've borrowed or moved a value.
3. We could allow writing `move` in patterns.
Without this, `move`, unlike `ref` and `ref mut`, would always be implicit,
leaving no way override a default binding mode of `ref` or `ref mut` and move
the value out from behind a reference.
However, moving a value out from behind a shared or mutable
reference is only possible for `Copy` types, so this would not be particularly
useful in practice, and would add unnecessary complexity to the language.
