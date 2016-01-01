- Feature Name: `drop_types_in_const`
- Start Date: 2016-01-01
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

Allow types with destructors to be used in `const`/`static` items, as long as the destructor is never run during `const` evaluation.

# Motivation
[motivation]: #motivation

Most collection types do not allocate any memory when constructed empty. With the change to make leaking safe, the restriction on `static` items with destructors
is no longer trequired to be a hard error.

Allowing types with destructors to be directly used in `const` functions and stored in `static`s will remove the need to have
runtime-initialisation for global variables.

# Detailed design
[design]: #detailed-design


- allow destructors in statics
 - optionally warn about the "potential leak"
- allow instantiating structures that impl Drop in constant expressions
- prevent const items from holding values with destructors, but allow const fn to return them
- disallow constant expressions which would result in the Drop impl getting called, where they not in a constant context

# Drawbacks
[drawbacks]: #drawbacks

Destructors do not run on `static` items (by design), so this can lead to unexpected behavior when a side-effecting type is stored in a `static` (e.g. a RAII temporary folder handle). However, this can already happen using the `lazy_static` crate.

# Alternatives
[alternatives]: #alternatives

- Runtime initialisation of a raw pointer can be used instead (as the `lazy_static` crate currently does on stable)
- On nightly, a bug related to `static` and `UnsafeCell<Option<T>>` can be used to remove the dynamic allocation.

Both of these alternatives require runtime initialisation, and incur a checking overhead on subsequent accesses.

# Unresolved questions
[unresolved]: #unresolved-questions

- TBD
