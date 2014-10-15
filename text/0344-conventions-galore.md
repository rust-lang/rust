- Start Date: 2014-10-15
- RFC PR: [rust-lang/rfcs#344](https://github.com/rust-lang/rfcs/pull/344)
- Rust Issue: [rust-lang/rust#18074](https://github.com/rust-lang/rust/issues/18074)

# Summary

This is a conventions RFC for settling a number of remaining naming conventions:

* Referring to types in method names
* Iterator type names
* Additional iterator method names
* Getter/setter APIs
* Associated types
* Trait naming
* Lint naming
* Suffix ordering
* Prelude traits

It also proposes to standardize on lower case error messages within the compiler
and standard library.

# Motivation

As part of the ongoing API stabilization process, we need to settle naming
conventions for public APIs. This RFC is a continuation of that process,
addressing a number of smaller but still global naming issues.

# Detailed design

The RFC includes a number of unrelated naming conventions, broken down into
subsections below.

## Referring to types in method names

Function names often involve type names, the most common example being conversions
like `as_slice`. If the type has a purely textual name (ignoring parameters), it
is straightforward to convert between type conventions and function conventions:

Type name | Text in methods
--------- | ---------------
`String`  | `string`
`Vec<T>`  | `vec`
`YourType`| `your_type`

Types that involve notation are less clear, so this RFC proposes some standard
conventions for referring to these types. There is some overlap on these rules;
apply the most specific applicable rule.

Type name | Text in methods
--------- | ---------------
`&str`    | `str`
`&[T]`    | `slice`
`&mut [T]`| `mut_slice`
`&[u8]`   | `bytes`
`&T`      | `ref`
`&mut T`  | `mut`
`*const T`| `ptr`
`*mut T`  | `mut_ptr`

The only surprise here is the use of `mut` rather than `mut_ref` for mutable
references. This abbreviation is already a fairly common convention
(e.g. `as_ref` and `as_mut` methods), and is meant to keep this very common case
short.

## Iterator type names

The current convention for iterator *type* names is the following:

> Iterators require introducing and exporting new types. These types should use
> the following naming convention:
>
> * **Base name**. If the iterator yields something that can be described with a
>    specific noun, the base name should be the pluralization of that noun
>    (e.g. an iterator yielding words is called `Words`). Generic contains use the
>    base name `Items`.
>
> * **Flavor prefix**. Iterators often come in multiple flavors, with the default
>   flavor providing immutable references. Other flavors should prefix their name:
>
>   * Moving iterators have a prefix of `Move`.
>   * If the default iterator yields an immutable reference, an iterator
>     yielding a mutable reference has a prefix `Mut`.
>   * Reverse iterators have a prefix of `Rev`.

(These conventions were established as part of
[this PR](https://github.com/rust-lang/rust/pull/8090) and later
[this one](https://github.com/rust-lang/rust/pull/11001).)

These conventions have not yet been updated to reflect the
[recent change](https://github.com/rust-lang/rfcs/pull/199) to the iterator
method names, in part to allow for a more significant revamp. There are some
problems with the current rules:

* They are fairly loose and therefore not mechanical or predictable. In
  particular, the choice of noun to use for the base name is completely
  arbitrary.

* They are not always applicable. The `iter` module, for example, defines a
  large number of iterator types for use in the adapter methods on `Iterator`
  (e.g. `Map` for `map`, `Filter` for `filter`, etc.) The module does not follow
  the convention, and it's not clear how it could do so.

This RFC proposes to instead align the convention with the `iter` module: the
name of an iterator type should be the same as the method that produces the
iterator.

For example:
* `iter` would yield an `Iter`
* `iter_mut` would yield an `IterMut`
* `into_iter` would yield an `IntoIter`

These type names make the most sense when prefixed with their owning module,
e.g. `vec::IntoIter`.

Advantages:

* The rule is completely mechanical, and therefore highly predictable.

* The convention can be (almost) universally followed: it applies equally well
  to `vec` and to `iter`.

Disadvantages:

* `IntoIter` is not an ideal name. Note, however, that since we've moved to
  `into_iter` as the method name, the existing convention (`MoveItems`) needs to
  be updated to match, and it's not clear how to do better than `IntoItems` in
  any case.

* This naming scheme can result in clashes if multiple containers are defined in
  the same module. Note that this is *already* the case with today's
  conventions.  In most cases, this situation should be taken as an indication
  that a more refined module hierarchy is called for.

## Additional iterator method names

An [earlier RFC](https://github.com/rust-lang/rfcs/pull/199) settled the
conventions for the "standard" iterator methods: `iter`, `iter_mut`,
`into_iter`.

However, there are many cases where you also want "nonstandard" iterator
methods: `bytes` and `chars` for strings, `keys` and `values` for maps,
the various adapters for iterators.

This RFC proposes the following convention:

* Use `iter` (and variants) for data types that can be viewed as containers,
  and where the iterator provides the "obvious" sequence of contained items.

* If there is no single "obvious" sequence of contained items, or if there are
  multiple desired views on the container, provide separate methods for these
  that do *not* use `iter` in their name. The name should instead directly
  reflect the view/item type being iterated (like `bytes`).

* Likewise, for iterator adapters (`filter`, `map` and so on) or other
  iterator-producing operations (`intersection`), use the clearest name to
  describe the adapter/operation directly, and do not mention `iter`.

* If not otherwise qualified, an iterator-producing method should provide an
  iterator over immutable references. Use the `_mut` suffix for variants
  producing mutable references, and the `into_` prefix for variants consuming
  the data in order to produce owned values.

## Getter/setter APIs

Some data structures do not wish to provide direct access to their fields, but
instead offer "getter" and "setter" methods for manipulating the field state
(often providing checking or other functionality).

The proposed convention for a field `foo: T` is:

* A method `foo(&self) -> &T` for getting the current value of the field.
* A method `set_foo(&self, val: T)` for setting the field. (The `val` argument
  here may take `&T` or some other type, depending on the context.)

Note that this convention is about getters/setters on ordinary data types, *not*
on [builder objects](http://aturon.github.io/ownership/builders.html). The
naming conventions for builder methods are still open.

## Associated types

Unlike type parameters, the *names* of
[associated types](https://github.com/rust-lang/rfcs/pull/195) for a trait are a
meaningful part of its public API.

Associated types should be given concise, but meaningful names, generally
following the convention for type names rather than generic. For example, use
`Err` rather than `E`, and `Item` rather than `T`.

## Trait naming

The wiki guidelines have long suggested naming traits as follows:

> Prefer (transitive) verbs, nouns, and then adjectives; avoid grammatical suffixes (like `able`)

Trait names like `Copy`, `Clone` and `Show` follow this convention. The
convention avoids grammatical verbosity and gives Rust code a distinctive flavor
(similar to its short keywords).

This RFC proposes to amend the convention to further say: if there is a single
method that is the dominant functionality of the trait, consider using the same
name for the trait itself. This is already the case for `Clone` and `ToCStr`,
for example.

According to these rules, `Encodable` should probably be `Encode`.

There are some open questions about these rules; see Unresolved Questions below.

## Lints

Our lint names are
[not consistent](https://github.com/rust-lang/rust/issues/16545). While this may
seem like a minor concern, when we hit 1.0 the lint names will be locked down,
so it's worth trying to clean them up now.

The basic rule is: the lint name should make sense when read as "allow
*lint-name*" or "allow *lint-name* items". For example, "allow
`deprecated` items" and "allow `dead_code`" makes sense, while "allow
`unsafe_block`" is ungrammatical (should be plural).

Specifically, this RFC proposes that:

* Lint names should state the bad thing being checked for,
  e.g. `deprecated`, so that `#[allow(deprecated)]` (items) reads
  correctly. Thus `ctypes` is not an appropriate name; `improper_ctypes` is.

* Lints that apply to arbitrary items (like the stability lints) should just
  mention what they check for: use `deprecated` rather than `deprecated_items`.
  This keeps lint names short. (Again, think "allow *lint-name* items".)

* If a lint applies to a specific grammatical class, mention that class and use
  the plural form: use `unused_variables` rather than `unused_variable`.
  This makes `#[allow(unused_variables)]` read correctly.

* Lints that catch unnecessary, unused, or useless aspects of code
  should use the term `unused`, e.g. `unused_imports`, `unused_typecasts`.

* Use snake case in the same way you would for function names.

## Suffix ordering

Very occasionally, conventions will require a method to have multiple suffixes,
for example `get_unchecked_mut`. When feasible, design APIs so that this
situation does not arise.

Because it is so rare, it does not make sense to lay out a complete convention
for the order in which various suffixes should appear; no one would be able to
remember it.

However, the *mut* suffix is so common, and is now entrenched as showing up in
final position, that this RFC does propose one simple rule: if there are
multiple suffixes including `mut`, place `mut` last.

## Prelude traits

It is not currently possible to define inherent methods directly on basic data
types like `char` or slices. Consequently, `libcore` and other basic crates
provide one-off traits (like `ImmutableSlice` or `Char`) that are intended to be
implemented solely by these primitive types, and which are included in the
prelude.

These traits are generally *not* designed to be used for generic programming,
but the fact that they appear in core libraries with such basic names makes it
easy to draw the wrong conclusion.

This RFC proposes to use a `Prelude` suffix for these basic traits. Since the
traits are, in fact, included in the prelude their names do not generally appear
in Rust programs. Therefore, choosing a longer and clearer name will help avoid
confusion about the intent of these traits, and will avoid namespace polution.

(There is one important drawback in today's Rust: associated functions in these
traits cannot yet be called directly on the types implementing the traits. These
functions are the one case where you would need to mention the trait by name,
today. Hopefully, this situation will change before 1.0; otherwise we may need a
separate plan for dealing with associated functions.)

## Error messages

Error messages -- including those produced by `fail!` and those placed in the
`desc` or `detail` fields of e.g. `IoError` -- should in general be in all lower
case. This applies to both `rustc` and `std`.

This is already the predominant convention, but there are some inconsistencies.

# Alternatives

## Iterator type names

The iterator type name convention could instead basically stick with today's
convention, but using suffixes instead of prefixes, and `IntoItems` rather than
`MoveItems`.

# Unresolved questions

How far should the rules for trait names go? Should we avoid "-er" suffixes,
e.g. have `Read` rather than `Reader`?
