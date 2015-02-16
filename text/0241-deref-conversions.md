- Start Date: 2014-09-16
- RFC PR: [rust-lang/rfcs#241](https://github.com/rust-lang/rfcs/pull/241)
- Rust Issue: [rust-lang/rust#21432](https://github.com/rust-lang/rust/issues/21432)

# Summary

Add the following coercions:

* From `&T` to `&U` when `T: Deref<U>`.
* From `&mut T` to `&U` when `T: Deref<U>`.
* From `&mut T` to `&mut U` when `T: DerefMut<U>`

These coercions eliminate the need for "cross-borrowing" (things like `&**v`)
and calls to `as_slice`.

# Motivation

Rust currently supports a conservative set of *implicit coercions* that are used
when matching the types of arguments against those given for a function's
parameters. For example, if `T: Trait` then `&T` is implicitly coerced to
`&Trait` when used as a function argument:

```rust
trait MyTrait { ... }
struct MyStruct { ... }
impl MyTrait for MyStruct { ... }

fn use_trait_obj(t: &MyTrait) { ... }
fn use_struct(s: &MyStruct) {
    use_trait_obj(s)    // automatically coerced from &MyStruct to &MyTrait
}
```

In older incarnations of Rust, in which types like vectors were built in to the
language, coercions included things like auto-borrowing (taking `T` to `&T`),
auto-slicing (taking `Vec<T>` to `&[T]`) and "cross-borrowing" (taking `Box<T>`
to `&T`).  As built-in types migrated to the library, these coercions have
disappeared: none of them apply today. That means that you have to write code
like `&**v` to convert `&Box<T>` or `Rc<RefCell<T>>` to `&T` and `v.as_slice()`
to convert `Vec<T>` to `&T`.

The ergonomic regression was coupled with a promise that we'd improve things in
a more general way later on.

"Later on" has come! The premise of this RFC is that (1) we have learned some
valuable lessons in the interim and (2) there is a quite conservative kind of
coercion we can add that dramatically improves today's ergonomic state of
affairs.

# Detailed design

## Design principles

### The centrality of ownership and borrowing

As Rust has evolved,
[a theme has emerged](http://blog.rust-lang.org/2014/09/15/Rust-1.0.html):
*ownership* and *borrowing* are the focal point of Rust's design, and the key
enablers of much of Rust's achievements.

As such, reasoning about ownership/borrowing is a central aspect of programming
in Rust.

In the old coercion model, borrowing could be done completely implicitly, so an
invocation like:

```rust
foo(bar, baz, quux)
```

might move `bar`, immutably borrow `baz`, and mutably borrow `quux`. To
understand the flow of ownership, then, one has to be aware of the details of
all function signatures involved -- it is not possible to see ownership at a
glance.

When
[auto-borrowing was removed](https://mail.mozilla.org/pipermail/rust-dev/2013-November/006849.html),
this reasoning difficulty was cited as a major motivator:

> Code readability does not necessarily benefit from autoref on arguments:

  ```rust
  let a = ~Foo;
  foo(a); // reading this code looks like it moves `a`
  fn foo(_: &Foo) {} // ah, nevermind, it doesn't move `a`!

  let mut a = ~[ ... ];
  sort(a); // not only does this not move `a`, but it mutates it!
  ```

Having to include an extra `&` or `&mut` for arguments is a slight
inconvenience, but it makes it much easier to track ownership at a glance.
(Note that ownership is not *entirely* explicit, due to `self` and macros; see
the [appendix](#appendix-ownership-in-rust-today).)

This RFC takes as a basic principle: **Coercions should never implicitly borrow from owned data**.

This is a key difference from the
[cross-borrowing RFC](https://github.com/rust-lang/rfcs/pull/226).

### Limit implicit execution of arbitrary code

Another positive aspect of Rust's current design is that a function call like
`foo(bar, baz)` does not invoke arbitrary code (general implicit coercions, as
found in e.g. Scala). It simply executes `foo`.

The tradeoff here is similar to the ownership tradeoff: allowing arbitrary
implicit coercions means that a programmer must understand the types of the
arguments given, the types of the parameters, and *all* applicable coercion code
in order to understand what code will be executed. While arbitrary coercions are
convenient, they come at a substantial cost in local reasoning about code.

Of course, method dispatch can implicitly execute code via `Deref`. But `Deref`
is a pretty specialized tool:

* Each type `T` can only deref to *one* other type.

  (Note: this restriction is not currently enforced, but will be enforceable
  once [associated types](https://github.com/rust-lang/rfcs/pull/195) land.)

* Deref makes all the methods of the target type visible on the source type.
* The source and target types are both references, limiting what the `deref`
  code can do.

These characteristics combined make `Deref` suitable for smart pointer-like
types and little else. They make `Deref` implementations relatively rare. And as
a consequence, you generally know when you're working with a type implementing
`Deref`.

This RFC takes as a basic principle: **Coercions should narrowly limit the code they execute**.

Coercions through `Deref` are considered narrow enough.

## The proposal

The idea is to introduce a coercion corresponding to `Deref`/`DerefMut`, but
*only* for already-borrowed values:

* From `&T` to `&U` when `T: Deref<U>`.
* From `&mut T` to `&U` when `T: Deref<U>`.
* From `&mut T` to `&mut U` when `T: DerefMut<U>`

These coercions are applied *recursively*, similarly to auto-deref for method
dispatch.

Here is a simple pseudocode algorithm for determining the applicability of
coercions.  Let `HasBasicCoercion(T, U)` be a procedure for determining whether
`T` can be coerced to `U` using today's coercion rules (i.e. without deref).
The general `HasCoercion(T, U)` procedure would work as follows:

```
HasCoercion(T, U):

  if HasBasicCoercion(T, U) then
      true
  else if T = &V and V: Deref<W> then
      HasCoercion(&W, U)
  else if T = &mut V and V: Deref<W> then
      HasCoercion(&W, U)
  else if T = &mut V and V: DerefMut<W> then
      HasCoercion(&W, U)
  else
      false
```

Essentially, the procedure looks for applicable "basic" coercions at increasing
levels of deref from the given argument, just as method resolution searches for
applicable methods at increasing levels of deref.

Unlike method resolution, however, this coercion does *not* automatically borrow.

### Benefits of the design

Under this coercion design, we'd see the following ergonomic improvements for
"cross-borrowing":

```rust
fn use_ref(t: &T) { ... }
fn use_mut(t: &mut T) { ... }

fn use_rc(t: Rc<T>) {
    use_ref(&*t);  // what you have to write today
    use_ref(&t);   // what you'd be able to write
}

fn use_mut_box(t: &mut Box<T>) {
    use_mut(&mut *t); // what you have to write today
    use_mut(t);       // what you'd be able to write

    use_ref(*t);      // what you have to write today
    use_ref(t);       // what you'd be able to write
}

fn use_nested(t: &Box<T>) {
    use_ref(&**t);  // what you have to write today
    use_ref(t);     // what you'd be able to write (note: recursive deref)
}
```

In addition, if `Vec<T>: Deref<[T]>` (as proposed
[here](https://github.com/rust-lang/rfcs/pull/235)), slicing would be automatic:

```rust
fn use_slice(s: &[u8]) { ... }

fn use_vec(v: Vec<u8>) {
    use_slice(v.as_slice());    // what you have to write today
    use_slice(&v);              // what you'd be able to write
}

fn use_vec_ref(v: &Vec<u8>) {
    use_slice(v.as_slice());    // what you have to write today
    use_slice(v);               // what you'd be able to write
}
```

### Characteristics of the design

The design satisfies both of the principles laid out in the Motivation:

* It does not introduce implicit borrows of owned data, since it only applies to
  already-borrowed data.

* It only applies to `Deref` types, which means there is only limited potential
  for implicitly running unknown code; together with the expectation that
  programmers are generally aware when they are using `Deref` types, this should
  retain the kind of local reasoning Rust programmers can do about
  function/method invocations today.

There is a *conceptual model* implicit in the design here: `&` is a "borrow"
operator, and richer coercions are available between borrowed types. This
perspective is in opposition to viewing `&` primarily as adding a layer of
indirection -- a view that, given compiler optimizations, is often inaccurate
anyway.

# Drawbacks

As with any mechanism that implicitly invokes code, deref coercions make it more
complex to fully understand what a given piece of code is doing. The RFC argued
inline that the design conserves local reasoning in practice.

As mentioned above, this coercion design also changes the mental model
surrounding `&`, and in particular somewhat muddies the idea that it creates a
pointer. This change could make Rust more difficult to learn (though note that
it puts *more* attention on ownership), though it would make it more convenient
to use in the long run.

# Alternatives

The main alternative that addresses the same goals as this RFC is the
[cross-borrowing RFC](https://github.com/rust-lang/rfcs/pull/226), which
proposes a more aggressive form of deref coercion: it would allow converting
e.g. `Box<T>` to `&T` and `Vec<T>` to `&[T]` directly. The advantage is even
greater convenience: in many cases, even `&` is not necessary. The disadvantage
is the change to local reasoning about ownership:

```rust
let v = vec![0u8, 1, 2];
foo(v); // is v moved here?
bar(v); // is v still available?
```

Knowing whether `v` is moved in the call to `foo` requires knowing `foo`'s
signature, since the coercion would *implicitly borrow* from the vector.

# Appendix: ownership in Rust today

In today's Rust, ownership transfer/borrowing is explicit for all
function/method arguments. It is implicit only for:

* *`self` on method invocations.* In practice, the name and context of a method
  invocation is almost always sufficient to infer its move/borrow semantics.

* *Macro invocations.* Since macros can expand into arbitrary code, macro
  invocations can appear to move when they actually borrow.
