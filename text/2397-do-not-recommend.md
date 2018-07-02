- Feature Name: `do_not_recommend`
- Start Date: 2018-04-07
- RFC PR: [rust-lang/rfcs#2397](https://github.com/rust-lang/rfcs/pull/2397)
- Rust Issue: [rust-lang/rust#51992](https://github.com/rust-lang/rust/issues/51992)

# Summary
[summary]: #summary

A new attribute can be placed on trait implementations: `#[do_not_recommend]`.
This attribute will cause the compiler to never recommend this impl transitively
as a way to implement another trait. For example, this would be placed on
`impl<T: Iterator> IntoIterator for T`. The result of this is that when `T:
IntoIterator` fails, the error message will only mention `IntoIterator`. It will
not say "perhaps `Iterator` should be implemented?".

# Motivation
[motivation]: #motivation

When a type fails to implement a trait, Rust has the wonderful behavior of
looking at possible *other* trait impls which might cause the trait in question
to be implemented. This is usually a good thing. For example, when using Diesel,
this is why instead of telling you `SelectStatement<{30 page long type}>:
ExecuteDsl is not satisifed`, it tells you `posts::id:
SelectableExpression<users::table> is not satisifed`.

However, there are times where this behavior actually makes the resulting error
more confusing. There are specific trait impls which almost always cause these
error messages to be more confusing. These are usually (but not always) very
broad blanket impls on traits with names like `IntoFoo` or `AsBar`. One such
problem impl is `impl<T: Iterator> IntoIterator for T`.

## `IntoIterator` confusion

Let's look at the struggles of a hypothetical Python programmer who is getting
into Rust for the first time. In Python, tuples are iterable. So our python
programmer writes this code expecting it to work:

```rust
for i in (1, 2, 3) {
    println!("{}", i);
}
```

They get the following error:

```
error[E0277]: the trait bound `({integer}, {integer}, {integer}): std::iter::Iterator` is not satisfied
 --> src/main.rs:2:14
  |
2 |     for x in (1, 2, 3) {
  |              ^^^^^^^^^ `({integer}, {integer}, {integer})` is not an iterator; maybe try calling `.iter()` or a similar method
  |
```

This error message is particularly bad for a failed `IntoIterator` constraint.
The only type in `std` which has a method called `iter` that doesn't implement
`IntoIterator` is a fixed sized array. For all of those types, it's generally
more idiomatic to just put an `&` in front of the value. And for this case,
neither one would be helpful even if it worked, since our hero is likely
expecting `x` to be `i32`, not `&i32`.

Following the advice of the error message, they try calling `.iter` on their
tuple, and get a new error:

```
error[E0599]: no method named `iter` found for type `({integer}, {integer}, {integer})` in the current scope
 --> src/main.rs:2:24
  |
2 |     for x in (1, 2, 3).iter() {
  |
```

At this point they remember a friend telling them they could see all of the
types that implement some trait in the docs. Tuples clearly aren't the type we
need, so let's see if we can find the type we *do* need. The error has told us
that we need to be looking at `Iterator`, so that's where we look in the docs.

The implementors section there is... less than helpful. Other than the type
`Map` (which our Rust newbie might incorrectly assume is `HashMap`), nothing
here looks helpful. It's mostly just weird types called `Iter` and weird
nonsense like `RSplitN`. At this point there's no obvious path to resolution.

If we had pointed them at `IntoIterator` like we should have, then the
implementors section... Well it actually wouldn't have been much more helpful,
since it's mostly just spammed with every single possible size of fixed sized
array. However, that's a completely separate problem, and at the very least vec
and slice, the type they most likely needed to see, are at least *somewhere* on
that page.

If nothing else, *in this particular case*, there was at least a note saying
"required by `std::iter::IntoIterator::into_iter`". However, the tiny footnote
at the bottom is not where most people look, and as we'll see later, is also not
always there or helpful.

## Ecosystem Examples

Let's look at another example from outside the standard library. This is a
problem Diesel has run into numerous times. The most common is with our
`AsExpression` trait. Diesel has a trait called `Expression`, which represents a
fragment of SQL with a known type. There is also a trait called `AsExpression`,
which is used to convert -- for example -- a Rust string into a data structure
representing a `TEXT` SQL expression. Unlike `IntoIterator`, where `Item` is an
associated type, in this case `SqlType` is a type parameter.

This gets represented in the type system to prevent things like accidentally
trying to compare a string with a text column. Problem code might look like
this: `a_table::id.eq(1)`. However, the error message they get is not so
helpful:

```
error[E0277]: the trait bound `str: diesel::Expression` is not satisfied
  --> src/lib.rs:14:17
   |
14 |     a_table::id.eq("1");
   |                 ^^ the trait `diesel::Expression` is not implemented for `str`
   |
   = note: required because of the requirements on the impl of `diesel::Expression` for `&str`
   = note: required because of the requirements on the impl of `diesel::expression::AsExpression<diesel::sql_types::Integer>` for `&str`
```

Even worse, since the body of `impl<T: Expression> AsExpression<T::SqlType> for
T` implies that the conversion returns `Self`, rust will continue on assuming
that `&str` is a type that appears in the final AST. This results in our less
than helpful message being even further behind 8 different trait impls that
would never be implemented for `&str` in the first place.

Once again, we do have this little foot note with the information we care about,
but as soon as we introduce one more layer of indirection, that gets completely
lost. For example, if that code were instead written as
`a_table::table.find("1")`, the full output we see is going to be:

```
error[E0277]: the trait bound `str: diesel::Expression` is not satisfied
  --> src/lib.rs:14:20
   |
14 |     a_table::table.find("1");
   |                    ^^^^ the trait `diesel::Expression` is not implemented for `str`
   |
   = note: required because of the requirements on the impl of `diesel::Expression` for `&str`
   = note: required because of the requirements on the impl of `diesel::Expression` for `diesel::expression::operators::Eq<a_table::columns::id, &str>`
   = note: required because of the requirements on the impl of `diesel::EqAll<&str>` for `a_table::columns::id`
   = note: required because of the requirements on the impl of `diesel::query_dsl::filter_dsl::FindDsl<&str>` for `a_table::table`

error[E0277]: the trait bound `str: diesel::expression::NonAggregate` is not satisfied
  --> src/lib.rs:14:20
   |
14 |     a_table::table.find("1");
   |                    ^^^^ the trait `diesel::expression::NonAggregate` is not implemented for `str`
   |
   = note: required because of the requirements on the impl of `diesel::expression::NonAggregate` for `&str`
   = note: required because of the requirements on the impl of `diesel::expression::NonAggregate` for `diesel::expression::operators::Eq<a_table::columns::id, &str>`
   = note: required because of the requirements on the impl of `diesel::query_dsl::filter_dsl::FilterDsl<diesel::expression::operators::Eq<a_table::columns::id, &str>>` for `diesel::query_builder::SelectStatement<a_table::table>`
```

Nowhere in this output is the *actual* missing trait (`AsExpression`) mentioned,
nor is the type parameter we care about (`sql_types::Integer`), which is *the
most important piece of information* ever mentioned.

The final motivation for this attribute is actually to *help* Rust give
transitive impls when it currently isn't. The only time Rust will recommend
implementing trait `T` in order to get an implementation of trait `U` is if
there is only one such impl which could potentially apply to your type that
would result in that behavior.

For example, Diesel has to provide a special impl to insert more than one row at
a time on SQLite, which doesn't have the keywords needed to safely do this in a
single query. However, on older versions of Diesel, if there is something
missing that causes that insert statement to not be valid, Rust will just give
up because it doesn't know if you wanted the "normal way to insert a thing" impl
to apply, or the "insert an iterator on SQLite" impl to apply. In the best case
this would result in "`InsertStatement<{30 page type}>: ExecuteDsl<Sqlite>` is
not satisfied", which is not helpful, but at least it's not actively misleading.
In the worst case it would result in "`YourRandomStruct: Iterator` is not
satisfied. Perhaps you need to implement it?" which is just complete nonsense.

With this annotation, Rust would know that it should *never* recommend the impl
related to `Iterators`, and will always give diagnostics as if the "normal way
to insert a thing" impl were the only one that existed.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

Since the diagnostics around this RFC aren't ever mentioned in a guide, I'm not
sure there would be a guide level explanation, but here goes:

Let's imagine you have the following traits:

```
pub trait Foo {
}

pub trait Bar {
}

impl<T: Foo> Bar for T {
}
```

If you tried to call a function that expects `T: Bar` with a type that does not
implement `Bar`, Rust will helpfully notice that if `T` implemented `Foo`, it
would also implement `Bar`. Because of that, it will recommend that you
implement `Foo` instead of `Bar`.

This is usually the desired behavior, but in some cases it can result in
confusing error messages. Perhaps when a function expects `Bar` and it's not
implemented, it would never make sense to implement `Foo` for that type. In this
case, we can put `#[do_not_recommend]` above our impl, and Rust will *never*
recommend implementing `Foo` as a way to get to `Bar`.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

During trait resolution, Rust will attempt to lower a query like
`IntoIterator(?T)` into a series of subqueries such as `IntoIterator(?T) :-
Iterator(?T)`. If only one such subquery exists, it will be used for error
diagnostics instead.

With this RFC, for the purposes of diagnostics only, impls annotated with
`#[do_not_recommend]` will be treated as if they did not exist. This means that
cases where there would have been one subquery will be treated as if there were
0, and cases where there were 2 will be treated as if there were 1.

# Drawbacks
[drawbacks]: #drawbacks

While this attribute only affects diagnostics, it is inherently tied to how
trait resolution works. This could potentially complicate work happening on the
trait system today (particularly with regards to chalk).

# Rationale and alternatives
[alternatives]: #alternatives

- The vast majority of cases where this would be used are for traits and impls
  that look very similar to `Iterator` and `impl<T: Iterator> IntoIterator for
  T`. We could potentially instead try to improve the compiler's diagnostics
  without this attribute, to detect those cases.

# Prior art
[prior-art]: #prior-art

The author is not aware of any prior art regarding this feature.

# Unresolved questions
[unresolved]: #unresolved-questions

- What other names could we go with besides `#[do_not_recommend]`?
