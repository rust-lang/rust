# `macro_literal_matcher`

The tracking issue for this feature is: [#35625]

The RFC is: [rfc#1576].

With this feature gate enabled, the [list of fragment specifiers][frags] gains one more entry:

* `literal`: a literal. Examples: 2, "string", 'c'

A `literal` may be followed by anything, similarly to the `ident` specifier.

[rfc#1576]: http://rust-lang.github.io/rfcs/1576-macros-literal-matcher.html
[#35625]: https://github.com/rust-lang/rust/issues/35625
[frags]: ../book/first-edition/macros.html#syntactic-requirements

------------------------
