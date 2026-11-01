# Nightly

This chapter documents style and formatting for nightly-only syntax. The rest of the style guide documents style for stable Rust syntax; nightly syntax only appears in this chapter. Each section here includes the name of the feature gate, so that searches (e.g. `git grep`) for a nightly feature in the Rust repository also turn up the style guide section.

Style and formatting for nightly-only syntax should be removed from this chapter and integrated into the appropriate sections of the style guide at the time of stabilization.

There is no guarantee of the stability of this chapter in contrast to the rest of the style guide. Refer to the style team policy for nightly formatting procedure regarding breaking changes to this chapter.

### Frontmatter

*Location: Placed before comments and attributes in the [root](index.html).*

*Tracking issue: [#136889](https://github.com/rust-lang/rust/issues/136889)*

*Feature gate: `frontmatter`*

There should be no blank lines between the frontmatter and either the start of the file or a shebang.
There can be zero or one line between the frontmatter and any following content.

The frontmatter fences should use the minimum number of dashes necessary for the contained content (one more than the longest series of initial dashes in the
content, with a minimum of 3 to be recognized as frontmatter delimiters).
If an infostring is present after the opening fence, there should be one space separating them.
The frontmatter fence lines should not have trailing whitespace.

```rust
#!/usr/bin/env cargo
--- cargo
[dependencies]
regex = "1"
---

fn main() {}
```
