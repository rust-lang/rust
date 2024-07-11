# Nightly

This chapter documents style and formatting for nightly-only syntax. The rest of the style guide documents style for stable Rust syntax; nightly syntax only appears in this chapter. Each section here includes the name of the feature gate, so that searches (e.g. `git grep`) for a nightly feature in the Rust repository also turn up the style guide section.

Style and formatting for nightly-only syntax should be removed from this chapter and integrated into the appropriate sections of the style guide at the time of stabilization.

There is no guarantee of the stability of this chapter in contrast to the rest of the style guide. Refer to the style team policy for nightly formatting procedure regarding breaking changes to this chapter.

### `feature(precise_capturing)`

A `use<'a, T>` precise capturing bound is formatted as if it were a single path segment with non-turbofished angle-bracketed args, like a trait bound whose identifier is `use`.

```
fn foo() -> impl Sized + use<'a> {}

// is formatted analogously to:

fn foo() -> impl Sized + Use<'a> {}
```
