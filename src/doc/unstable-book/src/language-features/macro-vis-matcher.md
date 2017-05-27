# `macro_vis_matcher`

The tracking issue for this feature is: [#41022]

With this feature gate enabled, the [list of fragment specifiers][frags] gains one more entry:

* `vis`: a visibility qualifier. Examples: nothing (default visibility); `pub`; `pub(crate)`.

A `vis` variable may be followed by a comma, ident, type, or path.

[#41022]: https://github.com/rust-lang/rust/issues/41022
[frags]: ../book/first-edition/macros.html#syntactic-requirements

------------------------
