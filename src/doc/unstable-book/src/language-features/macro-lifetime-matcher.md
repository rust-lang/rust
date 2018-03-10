# `macro_lifetime_matcher`

The tracking issue for this feature is: [#46895]

With this feature gate enabled, the [list of fragment specifiers][frags] gains one more entry:

* `lifetime`: a lifetime. Examples: 'static, 'a.

A `lifetime` variable may be followed by anything.

[#46895]: https://github.com/rust-lang/rust/issues/46895
[frags]: ../book/first-edition/macros.html#syntactic-requirements

------------------------
