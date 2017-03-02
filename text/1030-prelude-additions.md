- Feature Name: NA
- Start Date: 2015-04-03
- RFC PR: [rust-lang/rfcs#1030](https://github.com/rust-lang/rfcs/pull/1030)
- Rust Issue: [rust-lang/rust#24538](https://github.com/rust-lang/rust/issues/24538)

# Summary

Add `Default`, `IntoIterator` and `ToOwned` trait to the prelude.

# Motivation

Each trait has a distinct motivation:

* For `Default`, the ergonomics have vastly improved now that you can
  write `MyType::default()` (thanks to UFCS). Thanks to this
  improvement, it now makes more sense to promote widespread use of
  the trait.

* For `IntoIterator`, promoting to the prelude will make it feasible
  to deprecate the inherent `into_iter` methods and directly-exported
  iterator types, in favor of the trait (which is currently redundant).

* For `ToOwned`, promoting to the prelude would add a uniform,
  idiomatic way to acquire an owned copy of data (including going from
  `str` to `String`, for which `Clone` does not work).

# Detailed design

* Add `Default`, `IntoIterator` and `ToOwned` trait to the prelude.

* Deprecate inherent `into_iter` methods.

* Ultimately deprecate module-level `IntoIter` types (e.g. in `vec`);
  this may want to wait until you can write `Vec<T>::IntoIter` rather
  than `<Vec<T> as IntoIterator>::IntoIter`.

# Drawbacks

The main downside is that prelude entries eat up some amount of
namespace (particularly, method namespace). However, these are all
important, core traits in `std`, meaning that the method names are
already quite unlikely to be used.

Strictly speaking, a prelude addition is a breaking change, but as
above, this is highly unlikely to cause actual breakage. In any case,
it can be landed prior to 1.0.

# Alternatives

None.

# Unresolved questions

The exact timeline of deprecation for `IntoIter` types.

Are there other traits or types that should be promoted before 1.0?
